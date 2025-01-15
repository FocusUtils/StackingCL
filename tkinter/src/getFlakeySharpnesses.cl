int get_pos(int x, int y, int width, int color) {
    return (y * width + x) * 3 + color;
}



kernel void getFlakeySharpnesses(__global char *source,
                               __global double *flakey_sharpnesses,
                               int width, int height, int radius, float flake_size) {
    const int thrd_i = get_global_id(0);
    	
    float center_x = (int)((thrd_i + 1) * flake_size) % width;
    float center_y = ((int)((thrd_i) * flake_size) / width + 1) * flake_size;
    printf("thrd_i %i, center_x: %f, center_y: %f\n", thrd_i, center_x, center_y);

    if (center_x < flake_size || center_y < flake_size || center_x+flake_size > width || center_y+flake_size > height) {
        return;
    }
    for (int x = center_x - flake_size; x < center_x + flake_size; x++) {
        for (int y = center_y - flake_size; y < center_y + flake_size; y++) {
            if (x < 0 || y < 0 || x > width || y > height) {
                continue;
            }
            if (x == center_x && y == center_y) {
                continue;
            }
            
            flakey_sharpnesses[y * width + x] += (double)(thrd_i / 1000); 
        }
    }
}