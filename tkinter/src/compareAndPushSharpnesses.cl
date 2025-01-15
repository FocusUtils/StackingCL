int get_pos(int x, int y, int width, int color) {
    return (y * width + x) * 3 + color;
}

kernel void compareAndPushSharpnesses(__global char *destination,
                                      __global double *sharpnesses,
                                      __global char *source,
                                      int width, int height, int radius) {
    const int thrd_i = get_global_id(0);

    if (thrd_i > width * height) {
        return;
    }

    int center_x = thrd_i / height;
    int center_y = thrd_i % height;

    char center_b = source[get_pos(center_x, center_y, width, 0)];
    char center_g = source[get_pos(center_x, center_y, width, 1)];
    char center_r = source[get_pos(center_x, center_y, width, 2)];

    long delta = 0;

    int calculated_pixels = 0;

    for (int x = center_x - radius; x < center_x + radius + 1; x++) {
        for (int y = center_y - radius; y < center_y + radius + 1; y++) {
            if (x < 0 || y < 0 || x > width || y > height) {
                continue;
            }

            if (x == center_x && y == center_y) {
                continue;
            }

            float d =
                (float)(abs(abs(center_b) - abs(source[get_pos(x, y, width, 0)])) +
                        abs(abs(center_g) - abs(source[get_pos(x, y, width, 1)])) +
                        abs(abs(center_r) - abs(source[get_pos(x, y, width, 2)])));

            delta += (int)d;
            calculated_pixels++;
        }
    }

    double sharpness = (double)(delta) / (double)(calculated_pixels * 3 * 255);

    if (sharpness > sharpnesses[thrd_i]) {
        sharpnesses[thrd_i] = sharpness;
        destination[get_pos(center_x, center_y, width, 0)] = center_b;
        destination[get_pos(center_x, center_y, width, 1)] = center_g;
        destination[get_pos(center_x, center_y, width, 2)] = center_r;
    }
}

                               