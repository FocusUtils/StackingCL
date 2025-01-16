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

    float sharpness_in_patch = 0;
    for (int i = crcl_pos_start; i < crcl_pos_end; i += 2) {
        int x = circle_point_positions[i];
        int y = circle_point_positions[i + 1];
        if (x < 0 || x >= width || y < 0 || y >= height) {
            continue;
        }
        float sharpness_of_pixel = 0;
        for (int j = crcl_pos_start; j < crcl_pos_end; j += 2) {
            int x2 = circle_point_positions[j];
            int y2 = circle_point_positions[j + 1];
            if (x2 < 0 || x2 >= width || y2 < 0 || y2 >= height) {
                continue;
            }
            for (int color = 0; color < 3; color++) {
                sharpness_of_pixel += abs(source[get_pos(x, y, width, color)] - source[get_pos(x2, y2, width, color)]);
            }
        }
        sharpness_in_patch += sharpness_of_pixel / 3 / 255 / circle_point_positions_per_circle;
    }
    sharpness_in_patch /= circle_point_positions_per_circle;

    for (int i = crcl_pos_start; i < crcl_pos_end; i += 2) {
        int x = circle_point_positions[i];
        int y = circle_point_positions[i + 1];
        if (x < 0 || x >= width || y < 0 || y >= height) {
            continue;
        }
        flakey_sharpnesses[y * width + x] = sharpness_in_patch;

    }
}