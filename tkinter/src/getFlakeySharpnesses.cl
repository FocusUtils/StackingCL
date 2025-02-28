int get_pos(int x, int y, int width, int color) {
    return (y * width + x) * 3 + color;
}



kernel void getFlakeySharpnesses(__global uchar *source,
                               __global double *flakey_sharpnesses,
                               __global uchar* pixel_origins,
                               uchar source_index,
                               int width, int height, int radius) {
    const int thrd_i = get_global_id(0);

    if (thrd_i > width * height) {
        return;
    }

    int center_x = thrd_i / height;
    int center_y = thrd_i % height;

    uchar center_b = source[get_pos(center_x, center_y, width, 0)];
    uchar center_g = source[get_pos(center_x, center_y, width, 1)];
    uchar center_r = source[get_pos(center_x, center_y, width, 2)];

    long delta = 0;

    int calculated_pixels = 0;
    
    for (int x = center_x - radius; x < center_x + radius + 1; x++) {
        for (int y = center_y - radius; y < center_y + radius + 1; y++) {
            if (x < 0 || y < 0 || x >= width || y >= height) {
                continue;
            }

            if (x == center_x && y == center_y) {
                continue;
            }

            int b = abs(source[get_pos(x, y, width, 0)]);
            int g = abs(source[get_pos(x, y, width, 1)]);
            int r = abs(source[get_pos(x, y, width, 2)]);
            
            float d =
                (float)(abs(abs(center_b) - b) +
                        abs(abs(center_g) - g) +
                        abs(abs(center_r) - r));
            

            delta += (int)d;
            calculated_pixels++;
            
        }
    }
    
    
    double sharpness = (double)(delta) / (double)((int)calculated_pixels * 3 * 255);
    double brightness_normalized = (double)(center_b + center_g + center_r) / (double)(3 * 255);
    double sharpness_coefficient = -pow(brightness_normalized, .1) + 1;
    
    sharpness = sharpness * sharpness_coefficient;
    
    if (sharpness > flakey_sharpnesses[thrd_i]) {
        flakey_sharpnesses[thrd_i] = sharpness;
        pixel_origins[thrd_i] = source_index;
    }
}


kernel void pullPixelsByOriginImage(__global uchar *source,
                                    __global uchar *destination,
                                    __global uchar *pixel_origins,
                                    int width, int height, uchar source_index) {
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