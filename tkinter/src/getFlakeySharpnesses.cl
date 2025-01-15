int get_pos(int x, int y, int width, int color) {
    return (y * width + x) * 3 + color;
}



kernel void getFlakeySharpnesses(__global char *source,
                               __global double *flakey_sharpnesses,
                               int width, int height,
                               __global int* circle_point_positions,
                               int circle_point_positions_per_circle,
                               int circle_count) {
    const int thrd_i = get_global_id(0);
    if (thrd_i >= circle_count) {
        return;
    }
    int crcl_pos_start = thrd_i * circle_point_positions_per_circle * 2;
    int crcl_pos_end = crcl_pos_start + circle_point_positions_per_circle * 2; 

    printf("thread index: %d\n", thrd_i);

    for (int i = crcl_pos_start; i < crcl_pos_end; i += 2) {
        int x = circle_point_positions[i];
        int y = circle_point_positions[i + 1];
        if (x < 0 || x >= width || y < 0 || y >= height) {
            continue;
        }
        flakey_sharpnesses[y * width + x] = (double)(thrd_i % 99)/100;

    }
}