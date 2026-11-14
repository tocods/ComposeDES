template<int batch>
__global__ void paired_tiles(State s, block_q8_1 *other_hq, int producers, int gate_rows, int down_rows) {
    if (blockIdx.x < producers) {
        s.hq = other_hq;
        gate_tiles<batch, false>(s, blockIdx.x, producers, gate_rows);
    } else {
        down_tiles<batch, false>(s, blockIdx.x - producers, gridDim.x - producers, down_rows, 0);
    }
}
