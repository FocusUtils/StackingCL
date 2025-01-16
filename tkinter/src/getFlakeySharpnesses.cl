int get_pos(int x, int y, int width, int color) {
    return (y * width + x) * 3 + color;
}



kernel void getFlakeySharpnesses(__global char *source,
                               __global double *flakey_sharpnesses,
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

    if (sharpness > flakey_sharpnesses[thrd_i]) {
        flakey_sharpnesses[thrd_i] = sharpness;
    }
}

kernel void chooseOriginPixelBySharpnesses(__global double *all_sharpnesses,
                                            __global char* pixel_origins,
                                            int pixels_per_image,
                                            int image_count) {
    const int thrd_i = get_global_id(0);

    if (thrd_i > pixels_per_image) {
        return;
    }

    double max_sharpness = 0;
    int max_sharpness_index = 0;
    for (int i = 0; i < image_count; i++) {
        if (all_sharpnesses[i * pixels_per_image + thrd_i] > max_sharpness) {
            max_sharpness = all_sharpnesses[i * pixels_per_image + thrd_i];
            max_sharpness_index = i;
        }
    }
    pixel_origins[thrd_i] = max_sharpness_index;
}

kernel void pullPixelsByOriginImage(__global char *source,
                                    __global char *destination,
                                    __global char *pixel_origins,
                                    int width, int height, char source_index) {
    const int thrd_i = get_global_id(0);

    if (thrd_i > width * height) {
        return;
    }

    int x = thrd_i / height;
    int y = thrd_i % height;

    if (source_index == pixel_origins[thrd_i]) {
        destination[get_pos(x, y, width, 0)] = source[get_pos(x, y, width, 0)];
        destination[get_pos(x, y, width, 1)] = source[get_pos(x, y, width, 1)];
        destination[get_pos(x, y, width, 2)] = source[get_pos(x, y, width, 2)];
    }
}