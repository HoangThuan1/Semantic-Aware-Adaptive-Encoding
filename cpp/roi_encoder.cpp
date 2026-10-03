#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

struct Box { int left, top, right, bottom; };

static std::string av_error(int rc) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    av_strerror(rc, buf, sizeof(buf));
    return std::string(buf);
}

static void check(int rc, const std::string &what) {
    if (rc < 0) throw std::runtime_error(what + ": " + av_error(rc));
}

static std::map<int64_t, std::vector<Box>> load_rois(const std::string &path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open ROI schedule: " + path);
    std::map<int64_t, std::vector<Box>> result;
    std::string line;
    std::getline(in, line);  // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string field;
        std::vector<int64_t> values;
        while (std::getline(ss, field, ',')) values.push_back(std::stoll(field));
        if (values.size() != 5) throw std::runtime_error("invalid ROI CSV row: " + line);
        result[values[0]].push_back(Box{
            static_cast<int>(values[1]), static_cast<int>(values[2]),
            static_cast<int>(values[3]), static_cast<int>(values[4])});
    }
    return result;
}

struct Context {
    AVFormatContext *input = nullptr;
    AVFormatContext *output = nullptr;
    AVCodecContext *decoder = nullptr;
    AVCodecContext *encoder = nullptr;
    AVStream *input_stream = nullptr;
    AVStream *output_stream = nullptr;
    SwsContext *sws = nullptr;
    int video_stream = -1;
    int64_t frame_index = 0;
    int delta_qp = 0;
    std::map<int64_t, std::vector<Box>> rois;
};

static void drain_encoder(Context &c) {
    AVPacket *packet = av_packet_alloc();
    if (!packet) throw std::runtime_error("cannot allocate output packet");
    while (true) {
        int rc = avcodec_receive_packet(c.encoder, packet);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
        check(rc, "receive encoded packet");
        av_packet_rescale_ts(packet, c.encoder->time_base, c.output_stream->time_base);
        packet->duration = av_rescale_q(1, c.encoder->time_base, c.output_stream->time_base);
        packet->stream_index = c.output_stream->index;
        check(av_interleaved_write_frame(c.output, packet), "write encoded packet");
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
}

static void encode_frame(Context &c, const AVFrame *decoded) {
    AVFrame *frame = av_frame_alloc();
    if (!frame) throw std::runtime_error("cannot allocate encoder frame");
    frame->format = c.encoder->pix_fmt;
    frame->width = c.encoder->width;
    frame->height = c.encoder->height;
    frame->pts = c.frame_index;
    frame->color_range = AVCOL_RANGE_MPEG;
    frame->color_primaries = AVCOL_PRI_BT709;
    frame->color_trc = AVCOL_TRC_BT709;
    frame->colorspace = AVCOL_SPC_BT709;
    check(av_frame_get_buffer(frame, 32), "allocate encoder frame buffer");
    check(av_frame_make_writable(frame), "make encoder frame writable");

    sws_scale(c.sws, decoded->data, decoded->linesize, 0, decoded->height,
              frame->data, frame->linesize);

    const auto it = c.rois.find(c.frame_index);
    if (c.delta_qp != 0 && it != c.rois.end() && !it->second.empty()) {
        const size_t bytes = it->second.size() * sizeof(AVRegionOfInterest);
        AVFrameSideData *side = av_frame_new_side_data(
            frame, AV_FRAME_DATA_REGIONS_OF_INTEREST, bytes);
        if (!side) throw std::runtime_error("cannot allocate ROI side data");
        auto *roi = reinterpret_cast<AVRegionOfInterest *>(side->data);
        for (size_t i = 0; i < it->second.size(); ++i) {
            roi[i].self_size = sizeof(AVRegionOfInterest);
            roi[i].left = it->second[i].left;
            roi[i].top = it->second[i].top;
            roi[i].right = it->second[i].right;
            roi[i].bottom = it->second[i].bottom;
            roi[i].qoffset = AVRational{c.delta_qp, 51};
        }
    }

    check(avcodec_send_frame(c.encoder, frame), "send frame to encoder");
    av_frame_free(&frame);
    drain_encoder(c);
    ++c.frame_index;
}

static void drain_decoder(Context &c, AVFrame *decoded) {
    while (true) {
        int rc = avcodec_receive_frame(c.decoder, decoded);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) break;
        check(rc, "receive decoded frame");
        encode_frame(c, decoded);
        av_frame_unref(decoded);
    }
}

static void cleanup(Context &c) {
    if (c.output && !(c.output->oformat->flags & AVFMT_NOFILE) && c.output->pb)
        avio_closep(&c.output->pb);
    if (c.output) avformat_free_context(c.output);
    if (c.input) avformat_close_input(&c.input);
    avcodec_free_context(&c.decoder);
    avcodec_free_context(&c.encoder);
    sws_freeContext(c.sws);
}

int main(int argc, char **argv) {
    if (argc != 7) {
        std::cerr << "usage: roi_encoder INPUT.mkv ROI.csv OUTPUT.mp4 BITRATE_KBPS DELTA_QP GOP\n";
        return 2;
    }
    const std::string input_path = argv[1];
    const std::string roi_path = argv[2];
    const std::string output_path = argv[3];
    const int64_t bitrate = std::stoll(argv[4]) * 1000;
    const int delta_qp = std::stoi(argv[5]);
    const int gop = std::stoi(argv[6]);
    if (bitrate <= 0 || delta_qp > 0 || delta_qp < -51 || gop <= 0) {
        std::cerr << "invalid bitrate, delta QP, or GOP\n";
        return 2;
    }

    Context c;
    c.delta_qp = delta_qp;
    try {
        c.rois = load_rois(roi_path);
        check(avformat_open_input(&c.input, input_path.c_str(), nullptr, nullptr), "open input");
        check(avformat_find_stream_info(c.input, nullptr), "read input stream info");
        c.video_stream = av_find_best_stream(c.input, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
        check(c.video_stream, "find video stream");
        c.input_stream = c.input->streams[c.video_stream];

        const AVCodec *decoder = avcodec_find_decoder(c.input_stream->codecpar->codec_id);
        if (!decoder) throw std::runtime_error("decoder not found");
        c.decoder = avcodec_alloc_context3(decoder);
        if (!c.decoder) throw std::runtime_error("cannot allocate decoder context");
        check(avcodec_parameters_to_context(c.decoder, c.input_stream->codecpar), "copy decoder parameters");
        check(avcodec_open2(c.decoder, decoder, nullptr), "open decoder");

        check(avformat_alloc_output_context2(&c.output, nullptr, "mp4", output_path.c_str()), "create output");
        const AVCodec *encoder = avcodec_find_encoder_by_name("libx264");
        if (!encoder) throw std::runtime_error("required encoder libx264 not found");
        c.output_stream = avformat_new_stream(c.output, nullptr);
        if (!c.output_stream) throw std::runtime_error("cannot allocate output stream");
        c.encoder = avcodec_alloc_context3(encoder);
        if (!c.encoder) throw std::runtime_error("cannot allocate encoder context");

        c.encoder->width = c.decoder->width;
        c.encoder->height = c.decoder->height;
        c.encoder->pix_fmt = AV_PIX_FMT_YUV420P;
        c.encoder->time_base = AVRational{1, 30};
        c.encoder->framerate = AVRational{30, 1};
        c.encoder->bit_rate = bitrate;
        c.encoder->rc_max_rate = bitrate;
        c.encoder->rc_buffer_size = bitrate * 3;
        c.encoder->gop_size = gop;
        c.encoder->max_b_frames = 0;
        c.encoder->refs = 1;
        c.encoder->profile = FF_PROFILE_H264_HIGH;
        c.encoder->color_range = AVCOL_RANGE_MPEG;
        c.encoder->color_primaries = AVCOL_PRI_BT709;
        c.encoder->color_trc = AVCOL_TRC_BT709;
        c.encoder->colorspace = AVCOL_SPC_BT709;
        if (c.output->oformat->flags & AVFMT_GLOBALHEADER)
            c.encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        AVDictionary *options = nullptr;
        av_dict_set(&options, "preset", "veryfast", 0);
        av_dict_set(&options, "tune", "zerolatency", 0);
        av_dict_set(&options, "profile", "high", 0);
        std::ostringstream x264;
        x264 << "keyint=" << gop << ":min-keyint=" << gop
             << ":scenecut=0:bframes=0:ref=1:rc-lookahead=0:sliced-threads=0:threads=6"
             << ":nal-hrd=vbr:force-cfr=1:vbv-init=0.5:qpmin=10:qpmax=51:filler=0";
        av_dict_set(&options, "x264-params", x264.str().c_str(), 0);
        check(avcodec_open2(c.encoder, encoder, &options), "open libx264");
        av_dict_free(&options);
        check(avcodec_parameters_from_context(c.output_stream->codecpar, c.encoder), "copy encoder parameters");
        c.output_stream->time_base = c.encoder->time_base;

        if (!(c.output->oformat->flags & AVFMT_NOFILE))
            check(avio_open(&c.output->pb, output_path.c_str(), AVIO_FLAG_WRITE), "open output file");
        check(avformat_write_header(c.output, nullptr), "write output header");

        c.sws = sws_getContext(c.decoder->width, c.decoder->height, c.decoder->pix_fmt,
                               c.encoder->width, c.encoder->height, c.encoder->pix_fmt,
                               SWS_BICUBIC, nullptr, nullptr, nullptr);
        if (!c.sws) throw std::runtime_error("cannot create pixel conversion context");

        AVPacket *packet = av_packet_alloc();
        AVFrame *decoded = av_frame_alloc();
        if (!packet || !decoded) throw std::runtime_error("cannot allocate decoder objects");
        while (av_read_frame(c.input, packet) >= 0) {
            if (packet->stream_index == c.video_stream) {
                check(avcodec_send_packet(c.decoder, packet), "send input packet");
                drain_decoder(c, decoded);
            }
            av_packet_unref(packet);
        }
        check(avcodec_send_packet(c.decoder, nullptr), "flush decoder");
        drain_decoder(c, decoded);
        check(avcodec_send_frame(c.encoder, nullptr), "flush encoder");
        drain_encoder(c);
        check(av_write_trailer(c.output), "write trailer");
        av_frame_free(&decoded);
        av_packet_free(&packet);
        std::cerr << "encoded_frames=" << c.frame_index << "\n";
        cleanup(c);
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "roi_encoder: " << e.what() << "\n";
        cleanup(c);
        return 1;
    }
}
