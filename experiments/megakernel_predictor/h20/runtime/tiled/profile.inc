template<int batch, bool first>
__global__ void component(State s, int width) {
    if constexpr (first) gate_tiles<batch, false>(s, blockIdx.x, gridDim.x, width);
    else down_tiles<batch, false>(s, blockIdx.x, gridDim.x, width, 0);
}

template<int batch, bool first>
float time_component(State s, Timings &t, int blocks, int width, int size) {
    return t.measure([&] { component<batch, first><<<blocks, size, 0, t.stream>>>(s, width); });
}

template<int batch>
void measure(State &s, Timings &t, int nsm, const Data &frame) {
    auto flat = [&] {
        launch<batch, ROUTE>(t.stream, nsm, s, 256);
        launch<batch, XQ>(t.stream, nsm, s, 256);
    };
    flat();
    component<batch, true><<<nsm, 512, 0, t.stream>>>(s, 64);
    component<batch, false><<<nsm, 512, 0, t.stream>>>(s, 128);
    checked(cudaStreamSynchronize(t.stream));
    Routing rf{};
    checked(cudaMemcpy(&rf, s.routes, sizeof(rf), cudaMemcpyDeviceToHost));
    if (!std::equal(frame.ids.begin(), frame.ids.end(), rf.ids))
        throw std::runtime_error("captured expert IDs changed");
    const double coop = t.measure([&] { overhead<true, false>(t.stream, nsm); });
    const double plain = t.measure([&] { overhead<false, false>(t.stream, nsm); });
    const double barrier = t.measure([&] { overhead<true, true>(t.stream, nsm); });
    const double setup = t.measure([&] {
        launch<batch, ROUTE | XQ>(t.stream, nsm, s, 512);
    });
    const double reduction = t.measure([&] { launch<batch, REDUCE>(t.stream, nsm, s, 512); });
    std::string loads;
    for (int i = 0; i < rf.used; ++i) {
        if (i) loads += ':';
        loads += std::to_string(rf.groups[i].count);
    }
    for (int size : {256, 512}) {
        for (int blocks : {26, 39, 52, 78}) {
            if (blocks > nsm) continue;
            for (int rows : {32, 64, 128}) {
                const auto us = time_component<batch, true>(s, t, blocks, rows, size);
                std::printf("%d,%d,%d,%d,gate,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                            frame.batch, frame.layer, frame.cohort, frame.step, blocks, size, rows,
                            s.first_type, s.first_bytes, loads.c_str(), us, plain, coop, barrier, setup, reduction);
            }
            for (int rows : {64, 128, 256, 512}) {
                const auto us = time_component<batch, false>(s, t, blocks, rows, size);
                std::printf("%d,%d,%d,%d,down,%d,%d,%d,%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
                            frame.batch, frame.layer, frame.cohort, frame.step, blocks, size, rows,
                            s.down_type, s.down_bytes, loads.c_str(), us, plain, coop, barrier, setup, reduction);
            }
        }
    }
}
