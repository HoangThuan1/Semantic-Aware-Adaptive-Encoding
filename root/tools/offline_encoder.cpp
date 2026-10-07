// Host-only persistent libx264 reference. stdin: one target_kbps/delta_qp per GOP.
// stdout: actual payload bytes per GOP, so Python RL receives measured feedback.
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
extern "C" {
#include <x264.h>
}

struct Box { int x1, y1, x2, y2; };
int main(int argc, char **argv) {
    try {
        if (argc != 5) throw std::runtime_error("usage: encoder INPUT.nv12 ROI.csv OUTPUT.h264 cbr|vbr");
        std::ifstream input(argv[1], std::ios::binary), roi_file(argv[2]);
        std::ofstream output(argv[3], std::ios::binary);
        if (!input || !roi_file || !output) throw std::runtime_error("file open failed");
        std::map<int, std::vector<Box>> boxes;
        std::string line;
        std::getline(roi_file, line);
        while (std::getline(roi_file, line)) {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row(line);
            int frame; Box b;
            if (!(row >> frame >> b.x1 >> b.y1 >> b.x2 >> b.y2)) throw std::runtime_error("bad ROI CSV");
            boxes[frame].push_back(b);
        }
        x264_param_t param;
        if (x264_param_default_preset(&param, "veryfast", "zerolatency") < 0) throw std::runtime_error("preset failed");
        param.i_width=1920; param.i_height=1080; param.i_csp=X264_CSP_NV12;
        param.i_fps_num=30; param.i_fps_den=1; param.i_timebase_num=1; param.i_timebase_den=30;
        param.i_keyint_max=30; param.i_keyint_min=30; param.i_scenecut_threshold=0;
        param.i_bframe=0; param.i_frame_reference=1; param.rc.i_lookahead=0;
        // Frame threading buffers future frames and prevents exact per-GOP feedback.
        param.i_threads=1; param.b_sliced_threads=0; param.b_repeat_headers=1; param.b_annexb=1;
        param.vui.i_colorprim=1; param.vui.i_transfer=1; param.vui.i_colmatrix=1;
        param.rc.i_rc_method=X264_RC_ABR; param.rc.i_bitrate=1500;
        const bool cbr=std::string(argv[4])=="cbr";
        param.rc.i_vbv_max_bitrate=cbr ? 1500 : 8000;
        param.rc.i_vbv_buffer_size=cbr ? 4500 : 24000;
        param.rc.f_vbv_buffer_init=.5; param.rc.i_qp_min=10; param.rc.i_qp_max=51;
        // HRD fixes VBV signalling; disable it for dynamic VBR reconfiguration.
        param.i_nal_hrd=cbr ? X264_NAL_HRD_CBR : X264_NAL_HRD_NONE;
        param.rc.b_filler=cbr ? 1 : 0;
        if (x264_param_apply_profile(&param,"high") < 0) throw std::runtime_error("profile failed");
        x264_t *encoder=x264_encoder_open(&param);
        if (!encoder) throw std::runtime_error("x264 open failed");
        x264_picture_t pic, decoded;
        if (x264_picture_alloc(&pic, X264_CSP_NV12,1920,1080) < 0) throw std::runtime_error("picture alloc failed");
        std::vector<uint8_t> raw(1920*1080*3/2);
        std::vector<float> offsets(120*68);
        input.seekg(0, std::ios::end);
        const auto input_bytes=input.tellg();
        input.seekg(0);
        if (input_bytes<=0 || input_bytes%(30*1920*1080*3/2)!=0) throw std::runtime_error("input must contain whole GOPs");
        const int total_gops=static_cast<int>(input_bytes/(30*1920*1080*3/2));
        for (int gop=0; gop<total_gops; ++gop) {
            int bitrate, delta;
            if (!(std::cin >> bitrate >> delta) || bitrate<=0 || delta>0 || delta< -51) throw std::runtime_error("bad GOP command");
            x264_encoder_parameters(encoder,&param);
            param.rc.i_bitrate=bitrate;
            if (!cbr) {
                param.rc.i_vbv_max_bitrate=bitrate;
                param.rc.i_vbv_buffer_size=bitrate*3;
            } else if (bitrate!=1500 || delta!=0) throw std::runtime_error("CBR requires fixed 1500/no ROI");
            if (x264_encoder_reconfig(encoder,&param)<0) throw std::runtime_error("x264 reconfig failed");
            int64_t bytes=0;
            for (int j=0; j<30; ++j) {
                int frame=gop*30+j;
                if (!input.read(reinterpret_cast<char*>(raw.data()),raw.size())) throw std::runtime_error("short NV12 input");
                for (int y=0;y<1080;++y) std::copy_n(raw.data()+y*1920,1920,pic.img.plane[0]+y*pic.img.i_stride[0]);
                for (int y=0;y<540;++y) std::copy_n(raw.data()+1920*1080+y*1920,1920,pic.img.plane[1]+y*pic.img.i_stride[1]);
                std::fill(offsets.begin(),offsets.end(),0.f);
                for (const auto &b: boxes[frame])
                    for (int y=std::max(0,b.y1/16); y<std::min(68,(b.y2+15)/16); ++y)
                        for (int x=std::max(0,b.x1/16); x<std::min(120,(b.x2+15)/16); ++x)
                            offsets[y*120+x]=static_cast<float>(delta);
                pic.prop.quant_offsets=offsets.data(); pic.prop.quant_offsets_free=nullptr;
                pic.i_pts=frame; pic.i_type=j==0 ? X264_TYPE_IDR : X264_TYPE_P;
                x264_nal_t *nals; int n;
                int size=x264_encoder_encode(encoder,&nals,&n,&pic,&decoded);
                if (size<=0 || decoded.i_pts!=frame) throw std::runtime_error("unexpected x264 delay/frame mismatch");
                for (int i=0;i<n;++i) { output.write(reinterpret_cast<char*>(nals[i].p_payload),nals[i].i_payload); bytes+=nals[i].i_payload; }
            }
            output.flush();
            std::cout << bytes << std::endl;
        }
        if (x264_encoder_delayed_frames(encoder)!=0) throw std::runtime_error("unexpected pending frames");
        x264_picture_clean(&pic); x264_encoder_close(encoder);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}
