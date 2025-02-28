import customtkinter
from scrollable_frame import VerticalScrolledFrame
from preview_image import PreviewImage
from PIL import Image
import rawpy    
from tkinter import filedialog, messagebox, RIGHT, LEFT
import time
import gc
import os
import cv2
import numpy as np
from image_array_converter import convert_color_arr_to_image, convert_gray_arr_to_image, convert_gray_arr_to_gray_image
from math import sqrt
from threading import Thread
import multiprocess as mp
import subprocess
import sys
import statistics
import pyopencl as cl
from sbNative.runtimetools import get_path, exec_with_exc_tb
import traceback
import colorama
from sbNative.debugtools import log, ilog
import math
import json
if __name__ == "__main__":
    from lazyloading_image import LazyImage

class CustomSlider(customtkinter.CTkSlider):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.bind("<Enter>", self._enable_scroll)
        self.bind("<Leave>", self._disable_scroll)
        self.bind("<MouseWheel>", self._on_scroll)
        self._scroll_enabled = False
        self.on_scroll_events = set()
        self.on_value_update_events = set()
    
    def _enable_scroll(self, event):
        self._scroll_enabled = True
    
    def _disable_scroll(self, event):
        self._scroll_enabled = False
    
    def _on_scroll(self, event):
        if self._scroll_enabled:
            delta = 1 if event.delta > 0 else -1
            new_value = self.get() + delta
            self.set(min(max(new_value, self.cget("from_")), self.cget("to")))
            for callback in self.on_scroll_events:
                callback(self.get())
            
    def on_event_or_scroll(self, event, callback):
        self.bind(event, callback)
        self.on_scroll_events.add(callback)
        
    def on_value_update(self, callback):
        self.on_value_update_events.add(callback)
        
    def set(self, value):
        super().set(value)
        for callback in self.on_value_update_events:
            callback(value)


MULTIPLIER_GROW_BASE = 1.2
def normalize_sharpnesses(sharpness_gpu_original, width, height, multiplier=None):
    if multiplier is None:
        mean = np.mean(sharpness_gpu_original)
        multiplier = 127/mean
    sharpness_gpu = sharpness_gpu_original * multiplier
    sharpness_gpu[sharpness_gpu > 255] = 255
    return sharpness_gpu, multiplier
    


class ProgressBarMessage:
    def __init__(self, work_prefix, work_suffix, work_done, work_total, estimated_time_remaining):
        self.work_prefix = work_prefix
        self.work_suffix = work_suffix
        self.work_done = work_done
        self.work_total = work_total
        self.estimated_time_remaining = estimated_time_remaining

    def is_done(self):
        return self.work_done+1 == self.work_total

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


MAX_CORES_FOR_MP = mp.cpu_count()-1
ilog("mp cores assigned", MAX_CORES_FOR_MP)


FILE_EXTENTIONS = {
    "RAW": [
        ".nef",
        ".arw",
        ".raw",
    ],
    "CV2": [
        ".jpeg",
        ".jpg",
        ".png",
        ".tiff",
        ".tif"
    ],
}

def get_colortone(t):
    #       B                   G                   R
    return [255 * (1 - t),      80 * (t),    255 * t]


BLUE2ORANGE_LUT = np.zeros((256, 1, 3), dtype=np.uint8)
ORANGE2BLUE_LUT = np.zeros((256, 1, 3), dtype=np.uint8)
for i in range(256):
    
    
    t = i / 255.0  # Normalize
    t = 0.2 * math.tan(2.3 * (t - 0.5)) + 0.5
    BLUE2ORANGE_LUT[i, 0] = get_colortone(t)
    ORANGE2BLUE_LUT[i, 0] = get_colortone(1 - t)


def apply_lut_to_gray(gray, inverted=False):
    return cv2.LUT(cv2.merge([gray, gray, gray]), ORANGE2BLUE_LUT if inverted else BLUE2ORANGE_LUT)

def load_image(name):
    if any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["CV2"]):
        rgb = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)

    elif any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["RAW"]):
        with rawpy.imread(name) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=False)

    if rgb.shape[0] > rgb.shape[1]:
        
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)


    ## denoising
    return cv2.medianBlur(rgb, 1)


def save_image(name, cv2_image):
    if any(name.lower().endswith(ending) for ending in FILE_EXTENTIONS["CV2"]):
        cv2.imwrite(name, cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"Couldn't save the image due to not being able to save with the file type of {name}")


def find_nearest_pow_2(val):
    for i in range(32):
        if val < 2**i:
            return 2**i    


def get_opencl_devices():
    platforms = cl.get_platforms()  # Get all platforms
    return {device.name: device for platform in platforms for device in platform.get_devices()}
    


def initialize_gpu_and_compile(device: cl.Device):
    ctx = cl.Context([device])
    max_work_group_size = device.max_work_group_size
    queue = cl.CommandQueue(ctx)


    with open(get_path() / "getFlakeySharpnesses.cl", "r") as rf:
        source = rf.read()

    program = cl.Program(ctx, source).build()
    
    return ctx, max_work_group_size, program, queue


def render(radius, image_arr_dict, ctx, image_origin_manipulation_code, program, queue, message_queue):
    img1 = list(image_arr_dict.values())[0].rgb

    def get_estimated_pulling_time(calculating_sharpnesses_time):
        return (calculating_sharpnesses_time**(1/2.2) + .3 * len(image_arr_dict) - .1 * radius)
    
    width = int(img1.shape[1])
    height = int(img1.shape[0])
    total_pixels = width*height

    sharpness_gpu = np.zeros((total_pixels), dtype=np.float64)
    image_origin_gpu = np.zeros((total_pixels), dtype=np.uint8)

    mf = cl.mem_flags
    READ_WRITE = mf.READ_WRITE
    WRITE_ONLY = mf.WRITE_ONLY
    READ_ONLY = mf.READ_ONLY
    sharpnesses_buf = cl.Buffer(ctx, READ_WRITE | mf.COPY_HOST_PTR, hostbuf=sharpness_gpu)
    image_origin_buf = cl.Buffer(ctx, WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=image_origin_gpu)
    start_calculating_sharpnesses = time.time_ns()
    start_sharpness_and_origin_time = time.time_ns()
    for i, (name, lazyimage) in enumerate(image_arr_dict.items()):
        rgb = lazyimage.rgb
        bgr_flattened = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).flatten(order="K")
        lazyimage.cache()
        source_buf = cl.Buffer(ctx, READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bgr_flattened)
        del bgr_flattened
        try:

            # Execute the kernel
            program.getFlakeySharpnesses.set_scalar_arg_dtypes(
                [
                    None,
                    None,
                    None,
                    np.int8,
                    np.int32,
                    np.int32,
                    np.int32,
                ])
            
            program.getFlakeySharpnesses(
                queue, (total_pixels,), None,
                source_buf, sharpnesses_buf, image_origin_buf, np.int8(i), np.int32(width), np.int32(height), np.int32(radius)
            )


            # Wait for the operation to complete
            queue.finish()

            # Retrieve results from the GPU
            cl.enqueue_copy(queue, sharpness_gpu, sharpnesses_buf)
            cl.enqueue_copy(queue, image_origin_gpu, image_origin_buf)
        except:
            raise
        
        sharpness_and_origin_time_until_now = (time.time_ns() - start_sharpness_and_origin_time) / (10 ** 9)
        calculating_sharpnesses_time = (sharpness_and_origin_time_until_now/(i + 1)) * len(image_arr_dict)
        eta = get_estimated_pulling_time(calculating_sharpnesses_time) + calculating_sharpnesses_time - sharpness_and_origin_time_until_now
        message_queue.put(ProgressBarMessage("Calculating sharpnesses:", f"Image {name}", i, 2*len(image_arr_dict)+1, eta))
        
        
    calculating_sharpnesses_time = (time.time_ns() - start_calculating_sharpnesses) / (10 ** 9)
    

    try:
        # This is where the fun begins, the manipulation of the origin map, which decides which pixel to take from which image
        #
        message_queue.put(ProgressBarMessage("Manipulating origin map:", "", len(image_arr_dict), 2*len(image_arr_dict)+1, eta))
        image_origin_reshaped = image_origin_gpu.reshape((width, height))
        
        composite_image_gpu = np.zeros((total_pixels * 3), dtype=np.uint8)
        destination_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=composite_image_gpu)
        try:
            g = globals()
            l = locals()
            exec_with_exc_tb(image_origin_manipulation_code, g, l)
            image_origin_reshaped = l["image_origin_reshaped"]
        except Exception as e:
            print(colorama.Fore.RED + "Error in the origin manipulation code:")
            print(traceback.format_exc())
            print(colorama.Fore.RESET)
        
        start_pull_pixels_time = time.time_ns()
        image_origin_gpu = image_origin_reshaped.reshape(-1)
        del image_origin_reshaped
        image_origin_buf = cl.Buffer(ctx, READ_WRITE | mf.COPY_HOST_PTR, hostbuf=image_origin_gpu)

        
        for i, (name, lazyimage) in enumerate(image_arr_dict.items()):
            bgr_flattened = cv2.cvtColor(lazyimage.rgb, cv2.COLOR_RGB2BGR).flatten(order="K")
            lazyimage.cache()
            source_buf = cl.Buffer(ctx, READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bgr_flattened)
            program.pullPixelsByOriginImage.set_scalar_arg_dtypes([
                None,
                None,
                None,
                np.int32,
                np.int32,
                np.uint8,
            ])

            program.pullPixelsByOriginImage(
                queue, (total_pixels,), None,
                source_buf,
                destination_buf,
                image_origin_buf,
                np.int32(width),
                np.int32(height),
                np.uint8(i)
            )

            queue.finish()
            cl.enqueue_copy(queue, composite_image_gpu, destination_buf)
            pull_pixel_time_until_now = (time.time_ns() - start_pull_pixels_time) / (10 ** 9)
            eta = pull_pixel_time_until_now/(i + 1) * len(image_arr_dict) - pull_pixel_time_until_now
            message_queue.put(ProgressBarMessage("Pulling pixels by origin image:", f"Image {name}", len(image_arr_dict)+i+1, 2*len(image_arr_dict)+1, eta))

        
        return width, height, image_origin_gpu, composite_image_gpu, sharpness_gpu
    except:
        raise


if __name__ == '__main__':
    if sys.platform.startswith("win32"):
        mp.freeze_support()

    customtkinter.set_appearance_mode("dark")
    customtkinter.set_default_color_theme("dark-blue")

    root = customtkinter.CTk(fg_color="gray13")
    root.geometry("800x800")
    root.resizable(height=800, width=800)

    root.columnconfigure(0, weight=1)
    root.columnconfigure((1,2), weight=0)
    root.rowconfigure(0, weight=0)
    root.rowconfigure(1, weight=1)

    image_arr_dict = {}

    image_preview_frame = VerticalScrolledFrame(root)

    def add_image_to_scrollbar(img, name):
        container = customtkinter.CTkFrame(image_preview_frame.interior)

        target_diag_size = sqrt(2)*150

        curr_diag_size = sqrt(img.size[0]**2+img.size[1]**2)

        scaling_factor = curr_diag_size / target_diag_size
        tk_img = customtkinter.CTkImage(img, size=(img.size[0]//scaling_factor, img.size[1]//scaling_factor))
        img_panel = customtkinter.CTkLabel(container, image = tk_img, text="")
        
        name_panel = customtkinter.CTkLabel(container, text=name)


        def destroy_container():
            img_panel.pack_forget()
            destroy_button.pack_forget()
            container.pack_forget()
            name_panel.pack_forget()

            img_panel.destroy()
            destroy_button.destroy()
            container.destroy()
            name_panel.destroy()

            image_arr_dict.pop(name)

            gc.collect()

        destroy_button = customtkinter.CTkButton(container, text="Remove Image", command=destroy_container)
        img_panel.pack(padx=5, pady=5)
        name_panel.pack(padx=2, pady=2)
        destroy_button.pack(padx=2, pady=5)
        del img
        container.pack(padx=0, pady=5)


    def on_load_new_image():
        global loading_time
        selected_img_files = filedialog.askopenfiles(title="Open Images for the render queue", filetypes=[("Image-files", ".tiff .tif .png .jpg .jpeg .RAW .NEF")])
        if not selected_img_files:
            return

        img_load_time_start = time.time_ns()
        image_paths = []

        for f in selected_img_files:
            if os.path.basename(f.name) in image_arr_dict.keys():
                continue
            
            image_paths.append(f.name)

        if len(image_paths) == 0:
            return
        rgb_values = mp.Pool(min(MAX_CORES_FOR_MP, len(image_paths))).imap(load_image, image_paths)
        start_image_load_time = time.time_ns()
        for idx, (name, rgb) in enumerate(zip(image_paths, rgb_values)):
            img = Image.fromarray(cv2.resize(rgb, (int(rgb.shape[1]//5), int(rgb.shape[0]//5))))
            add_image_to_scrollbar(img, os.path.basename(name))

            image_arr_dict[os.path.basename(name)] = LazyImage(rgb, name)
            del img
            gc.collect()
            eta = (time.time_ns() - start_image_load_time) / (10 ** 9) / (idx + 1) * (len(image_paths) - idx)
            message_queue.put(ProgressBarMessage("Loading images:", f"Image {os.path.basename(name)}", idx, len(image_paths), eta))
        loading_time = (time.time_ns() - img_load_time_start) / (10 ** 9) 
        


    global output_panel
    global changes_panel
    global sharpness_panel

    global output_panel_packed
    global changes_panel_packed
    global sharpness_panel_packed

    global changes_img
    global output_img
    global sharpness_img

    global radius
    
    global sharpnesses_gpu
    global changes_arr
    global rendering_time
    global loading_time
    

    output_panel = None
    output_panel_packed = False
    output_img = None

    changes_panel = None
    changes_panel_packed = False
    changes_img = None

    sharpness_panel = None
    sharpness_panel_packed = None
    sharpness_img = None
    radius = 1

    sharpnesses_gpu = None
    image_origin_gpu = None
    rendering_time = -1
    loading_time = -1

    global image_origin_manipulation_code
    image_origin_manipulation_code = "image_origin_reshaped = image_origin_reshaped"

    
    def zoom_event_callback(current_panel):
        for other_panel in [output_panel, changes_panel, sharpness_panel]:
            if other_panel is current_panel:
                continue
            other_panel.zoom_amount = current_panel.zoom_amount
            other_panel.zoom_x_offset = current_panel.zoom_x_offset
            other_panel.zoom_y_offset = current_panel.zoom_y_offset
            other_panel.redraw_image()
            
    

    def pack_img_panel():
        global output_panel
        global output_panel_packed
        if output_panel:
            output_panel.pack(padx=5, pady=5, expand=True, fill = "both")
            output_panel_packed = True


    def pack_changes_panel():
        global changes_panel
        global changes_panel_packed
        if changes_panel:
            changes_panel.pack(padx=5, pady=5, expand=True, fill = "both")
            changes_panel_packed = True


    def pack_sharpness_panel():
        global sharpness_panel
        global sharpness_panel_packed
        if sharpness_panel:
            sharpness_panel.pack(padx=5, pady=5, expand=True, fill = "both")
            sharpness_panel_packed = True


    def unpack_img_panel():
        global output_panel
        global output_panel_packed
        output_panel_packed = False
        if output_panel:
            output_panel.pack_forget()
            return 1
        return 0


    def unpack_changes_panel():
        global changes_panel
        global changes_panel_packed
        changes_panel_packed = False
        if changes_panel:
            changes_panel.pack_forget()
            return 1
        return 0


    def unpack_sharpness_panel():
        global sharpness_panel
        global sharpness_panel_packed
        sharpness_panel_packed = False
        if sharpness_panel:
            sharpness_panel.pack_forget()
            return 1
        return 0

    
    def launch_render():
        if len(image_arr_dict) < 2:
            messagebox.showerror("Render exception", "Exception: You have not opened 2 or more images to the render queue.")
            return
        device = [device for device in list(get_opencl_devices().values()) if device.name in gpu_selection_dropdown.get()][0]
        ctx, max_work_group_size, program, queue = initialize_gpu_and_compile(device)
        global changes_panel
        global output_panel
        global sharpness_panel
        global changes_img
        global output_img
        global sharpness_img
        global radius

        global sharpnesses_gpu
        global changes_arr
        global rendering_time
        
        render_time_start = time.time_ns()
        width, height, changes_arr, composite_image_gpu, sharpnesses_gpu = render(radius, image_arr_dict, ctx, image_origin_manipulation_code, program, queue, message_queue)
        
        
        rendering_time = (time.time_ns() - render_time_start) / (10 ** 9)

        unpack_img_panel()
        unpack_changes_panel()
        unpack_sharpness_panel()
        
        rendered_images_frame = customtkinter.CTkFrame(rendering_frame, fg_color="gray13")
        rendered_images_frame.grid(row=1, column=0, columnspan=3, sticky="nesw")
        

        changes_img = apply_lut_to_gray(convert_gray_arr_to_gray_image(changes_arr * int(255 / len(image_arr_dict)), width, height), inverted=True)
        changes_panel = PreviewImage(rendered_images_frame, update_img_pos_info_strvar, image = changes_img)
        changes_panel.add_zoom_event_callback(zoom_event_callback)
        on_show_changes_checkbox()

        output_img = convert_color_arr_to_image(composite_image_gpu, width, height)
        output_panel = PreviewImage(rendered_images_frame, update_img_pos_info_strvar, image = output_img)
        output_panel.add_zoom_event_callback(zoom_event_callback)
        on_show_output_checkbox()
        
        
        brightnessed, multiplier = normalize_sharpnesses(sharpnesses_gpu, width, height)
        sharpness_brightness_slider.set(math.log(multiplier, MULTIPLIER_GROW_BASE))
        
        sharpness_img = convert_gray_arr_to_image(brightnessed, width, height)
        
        sharpness_panel = PreviewImage(rendered_images_frame, update_img_pos_info_strvar, image = sharpness_img)
        
        sharpness_panel.add_zoom_event_callback(zoom_event_callback)
        on_show_sharpness_checkbox()


    ## worker bar
    def initialize_progress(name, display_info=True):
        progress_label_strvar.set(name)
        progress_info_strvar.set("")
        progress_bar.set(0)

        progress_label.pack(padx=10, side=LEFT)
        progress_bar.pack(padx=10, side=LEFT)
        if display_info:
            progress_info.pack(padx=10, side=RIGHT)
    

    def deinitialize_progress():
        progress_label_strvar.set("")
        progress_info_strvar.set("")
        progress_bar.set(0)
        

        progress_label.pack_forget()
        progress_bar.pack_forget()
        progress_info.pack_forget()


    worker_frame = customtkinter.CTkFrame(root, height=30, fg_color="Black")
    worker_frame.grid(row=0, column=0, columnspan=3, padx=20, pady=10)

    progress_label_strvar = customtkinter.StringVar(value="a progress:")
    progress_label = customtkinter.CTkLabel(worker_frame, textvariable=progress_label_strvar)

    progress_bar = customtkinter.CTkProgressBar(worker_frame)

    progress_info_strvar = customtkinter.StringVar(value="10/50")
    progress_info = customtkinter.CTkLabel(worker_frame, textvariable=progress_info_strvar)


    ## rendering

    rendering_frame = customtkinter.CTkFrame(root)
    rendering_frame.grid(row=1, column=0, sticky="nesw")
    rendering_frame.columnconfigure(0, weight=1)
    rendering_frame.columnconfigure(1, weight=0)
    rendering_frame.columnconfigure(2, weight=1)
    rendering_frame.rowconfigure(0, weight=0)
    rendering_frame.rowconfigure(1, weight=1)

    render_button = customtkinter.CTkButton(rendering_frame, text="Render opened images",
                                            command=lambda: Thread(target=launch_render).start())
    render_button.grid(row=0, column=0, pady=(12, 5), padx = (0, 12), sticky="ne")
    
    
    gpu_selection_label = customtkinter.CTkLabel(rendering_frame, text="Select a GPU:",
                                                 fg_color=render_button._fg_color,
                                                 corner_radius=render_button._corner_radius)
    gpu_selection_label.grid(row=0, column=1, pady=(12, 5), padx=(1, 1), sticky="n")
    
    ## create a dropdown for selecting a gpu
    
    cl_devices = get_opencl_devices()
    
    options = [f"{name} ({device.vendor})" for name, device in cl_devices.items()]
    if len(options) == 0:
        messagebox.showerror("No OpenCL devices found", "Error: No OpenCL devices were found on your system. Please make sure you have OpenCL installed for the GPU you intend to use and your drivers are up to date.")
    gpu_selection_dropdown = customtkinter.CTkComboBox(rendering_frame, values=options)
    gpu_selection_dropdown.set(options[0])  # Set the default value
    gpu_selection_dropdown.grid(row=0, column=2, pady=(12, 5), padx=(0, 0), sticky="nw")


    ## settings

    settings_frame = customtkinter.CTkFrame(root, width=100)
    settings_frame.grid(row=1, column=1, sticky="nesw")


    def on_show_changes_checkbox():
        if show_changes_intvar.get() == 1:
            pack_changes_panel()
        else:
            unpack_changes_panel()

    show_changes_intvar = customtkinter.IntVar(value=1)
    show_changes_checkbox = customtkinter.CTkCheckBox(settings_frame, text="Show Changes Image", variable=show_changes_intvar,
                                                    onvalue=1, offvalue=0, command=on_show_changes_checkbox)
    show_changes_checkbox.grid(pady=5, row=2, column=0, sticky="nw")


    def on_show_output_checkbox():
        if show_output_intvar.get() == 1:
            pack_img_panel()
        else:
            unpack_img_panel()

    show_output_intvar = customtkinter.IntVar(value=1)
    show_output_checkbox = customtkinter.CTkCheckBox(settings_frame, text="Show Output Image", variable=show_output_intvar,
                                                    onvalue=1, offvalue=0, command=on_show_output_checkbox)
    show_output_checkbox.grid(pady=5, row=3, column=0, sticky="nw")


    def on_show_sharpness_checkbox():
        if show_sharpness_intvar.get() == 1:
            pack_sharpness_panel()
        else:
            unpack_sharpness_panel()

    show_sharpness_intvar = customtkinter.IntVar(value=1)
    show_sharpness_checkbox = customtkinter.CTkCheckBox(settings_frame, text="Show Sharpness Image", variable=show_sharpness_intvar,
                                                    onvalue=1, offvalue=0, command=on_show_sharpness_checkbox)
    show_sharpness_checkbox.grid(pady=5, row=4, column=0, sticky="nw")

    image_position_info_strvar = customtkinter.StringVar(value="No image loaded")
    image_position_info = customtkinter.CTkLabel(settings_frame, textvariable=image_position_info_strvar)
    image_position_info.grid(pady=5, row=6, column=0, sticky="nw")
    
    def on_sharpness_brightness_slider_value_update(value):
        multiplier = MULTIPLIER_GROW_BASE**sharpness_brightness_slider.get()
        order_of_magnitude = int(math.log(multiplier, 10))
        multiplier_scientific = f"{multiplier/10**order_of_magnitude:.2f}e{order_of_magnitude}"
        sharpness_brightness_string_var.set(f"Sharpness brightness: {multiplier_scientific}")
    
    def on_sharpness_brightness_slider_change(*_):
        global sharpness_img
        global sharpnesses_gpu
        multiplier = MULTIPLIER_GROW_BASE**sharpness_brightness_slider.get()
        if sharpnesses_gpu is None:
            return
        brightnessed, _ = normalize_sharpnesses(sharpnesses_gpu, sharpness_img.shape[1], sharpness_img.shape[0], multiplier)
        sharpness_img = convert_gray_arr_to_image(brightnessed, sharpness_img.shape[1], sharpness_img.shape[0])
        sharpness_panel.update_image(sharpness_img)
    
    ## sharpness brightness slider
    sharpness_brightness_string_var = customtkinter.StringVar(value="Sharpness brightness: 1.00e0")
    sharpness_brightness_label = customtkinter.CTkLabel(settings_frame, textvariable=sharpness_brightness_string_var)
    sharpness_brightness_label.grid(pady=5, row=8, column=0, sticky="nw")
    sharpness_brightness_slider = CustomSlider(settings_frame, from_=0, to=200)
    sharpness_brightness_slider.set(0)
    sharpness_brightness_slider.grid(pady=5, row=9, column=0, sticky="nw")
    sharpness_brightness_slider.on_event_or_scroll("<ButtonRelease-1>", on_sharpness_brightness_slider_change)
    sharpness_brightness_slider.on_value_update(on_sharpness_brightness_slider_value_update)
    
    
    def update_img_pos_info_strvar(x, y):
        image_position_info_strvar.set(f"Mouse Position: ({x}, {y})")

    global radius_string_var
    def on_radius_slider(event):
        global radius
        global radius_string_var
        radius = int(radius_slider.get())
        radius_string_var.set(f"Radius: {radius}")

    radius_string_var = customtkinter.StringVar(value="Radius: 1")
    radius_label = customtkinter.CTkLabel(settings_frame, textvariable=radius_string_var)
    radius_label.grid(pady=0, row=0, column=0, sticky="s")
    radius_slider = CustomSlider(settings_frame, from_=1, to=60)
    radius_slider.on_event_or_scroll("<ButtonRelease-1>", on_radius_slider)
    radius_slider.set(1)
    radius_slider.grid(pady=5, row=1, column=0, sticky="nw")

    origin_manipulation_textbox = customtkinter.CTkTextbox(settings_frame, text_color="white")
    origin_manipulation_textbox.grid(pady=5, row=7, column=0, sticky="nw")
    origin_manipulation_textbox.insert(1.0, image_origin_manipulation_code)

    def on_origin_manipulation_textbox_change(event):
        global image_origin_manipulation_code
        image_origin_manipulation_code = origin_manipulation_textbox.get(1.0, "end-1c")
        try:
            compile(image_origin_manipulation_code, "<string>", "exec")
        except SyntaxError:
            origin_manipulation_textbox.configure(text_color="red")
        else:
            origin_manipulation_textbox.configure(text_color="white")

    
    origin_manipulation_textbox.bind("<KeyRelease>", on_origin_manipulation_textbox_change)
    
    
    def on_save_selected_button():
        global output_img
        global output_panel_packed
        global changes_img
        global changes_panel_packed
        global sharpness_img
        global sharpness_panel_packed

        file_name = filedialog.asksaveasfilename(title="Save as filenames", defaultextension=".png", filetypes=[("PNG", ".png"), ("JPG", ".jpg"), ("TIFF", ".tiff")])

        exported_img = False
        exported_chng = False
        exported_shrp = False
        if output_panel_packed and output_img is not None:
            save_image(file_name, output_img)
            exported_img = True

        if changes_panel_packed and changes_img is not None:
            save_image(".".join(file_name.split(".")[:-1] + ["changes"] + [file_name.split(".")[-1]]), changes_img)
            exported_chng = True

        if sharpness_panel_packed and sharpness_img is not None:
            save_image(".".join(file_name.split(".")[:-1] + ["sharpnesses"] + [file_name.split(".")[-1]]), sharpness_img)
            exported_shrp = True

        
        ## metadata
        if not (exported_img or exported_chng or exported_shrp):
            messagebox.showwarning(title="Export warning", message="Warning: Nothing was saved because you have either not rendered the images yet or you unchecked every option above!")
            return
        else:
            ## statistics
            statistics_calc_time_start = time.time_ns()

            meta_data_lst = [
                                f'"Radius":                     {radius}X{radius}px mesh\n\n'
                                f"Exported output image:        {exported_img}\n",
                                f"Exported sharpness map:       {exported_shrp}\n",
                                f"Exported changes map:         {exported_chng}\n\n",

                                f"Min sharpness:                {np.amin(sharpnesses_gpu):.15f} / 1\n",
                                f"Max sharpness:                {np.amax(sharpnesses_gpu):.15f} / 1\n",

                                f"Average sharpness:            {np.average(sharpnesses_gpu):.15f} / 1\n",
                                f"Median sharpness:             {np.median(sharpnesses_gpu):.15f} / 1\n",
                                f"Mean sharpness:               {np.mean(sharpnesses_gpu):.15f} / 1\n\n",
                            ]
            
            pxl_nums = []
            for number, count in dict(zip(*np.unique(changes_arr, return_counts=True))).items():
                if number == 0:
                    continue
                pxl_nums.append(count)
                meta_data_lst.append(
                    f"Pixels used from image {list(image_arr_dict.keys())[number-1]}: \t{count:{len(str(output_img.shape[0]*output_img.shape[1]))}d} ({(count / (len(changes_arr)-1) * 100):.2f}%)\n")

            meta_data_lst.append("\n")
            
            min_pxls_used = min(pxl_nums)
            max_pxls_used = max(pxl_nums)
                                   
            meta_data_lst.append(f"Min count of pixels used:     {min_pxls_used} ({(min_pxls_used / (len(changes_arr)-1) * 100):.2f}%)\n")
            meta_data_lst.append(f"Max count of pixels used:     {max_pxls_used} ({(max_pxls_used / (len(changes_arr)-1) * 100):.2f}%)\n")

            avg_pxls_used = statistics.mean(pxl_nums)
            mdn_pxls_used = statistics.median(pxl_nums)
            meta_data_lst.append(f"Average count of pixels used: {avg_pxls_used} ({(avg_pxls_used / (len(changes_arr)-1) * 100):.2f}%)\n")
            meta_data_lst.append(f"Median count of pixels used:  {mdn_pxls_used} ({(mdn_pxls_used / (len(changes_arr)-1) * 100):.2f}%)\n")


            ## meta-meta statistics
            statistics_delta_time = (time.time_ns() - statistics_calc_time_start) / (10 ** 9)
            meta_data_lst.append("\n\n")
            meta_data_lst.append(f"Loading images took           {loading_time:.5f} seconds\n")
            meta_data_lst.append(f"Rendering images took         {rendering_time:.5f} seconds\n")

            meta_data_lst.append(f"Computing statistics took     {statistics_delta_time:.5f} seconds\n")

            meta_data_file_name = ".".join(file_name.split(".")[:-1] + ["metadata.txt"])

            try:
                os.remove(meta_data_file_name)
            except OSError:
                pass
            with open(meta_data_file_name, "w") as wf_meta_data:
                wf_meta_data.writelines(meta_data_lst)

                
        if os.name == "nt":
            win_style_dir = file_name.replace("/", "\\")
            cmd = f'explorer /select,"{win_style_dir}"'
            subprocess.Popen(cmd)
        


    save_selected_button = customtkinter.CTkButton(settings_frame, text="Save shown images", command=on_save_selected_button)
    save_selected_button.grid(pady=5, row=5, column=0, sticky="nw")

    ## previews

    def launch_on_load_new_image():
        Thread(target=on_load_new_image).start()

    image_preview_frame.grid(row=1, column=2, ipadx=20, sticky="nesw")
    load_new_image_button = customtkinter.CTkButton(master=image_preview_frame.interior,
                                                    text="Load new image", command=launch_on_load_new_image)
    load_new_image_button.pack(pady=(12, 5))

    def update_progress_bar_worker(message_queue):
        message = None
        target_finish_time = -1
        shown_finish_time = -1
        initialized = False
        while True:
            try:
                message = message_queue.get_nowait()
                if message.estimated_time_remaining != -1:
                    target_finish_time = message.estimated_time_remaining + time.time()
                    if shown_finish_time == -1:
                        shown_finish_time = target_finish_time
            except:
                pass
            time.sleep(.1)
            shown_finish_time += (target_finish_time - shown_finish_time) * .9
            if message is None:
                continue

            if message.is_done():
                deinitialize_progress()
                initialized = False
                shown_finish_time = -1
                continue

            if not initialized or message.work_prefix != progress_label_strvar.get():
                initialize_progress(message.work_prefix)
                initialized = True

            progress_bar.set((message.work_done+1)/message.work_total)
            suffix_texts = ["("]
            if message.work_total != -1:
                suffix_texts.append(f"{message.work_done+1}/{message.work_total}")
            if message.work_suffix:
                suffix_texts.append(message.work_suffix)
            if target_finish_time != -1:
                suffix_texts.append(f"ETA: {shown_finish_time - time.time():.2f}s")
            suffix_texts.append(")")
            progress_info_strvar.set(" ".join(suffix_texts))
        
        

    manager = mp.Manager()
    message_queue = manager.Queue()
    Thread(target=update_progress_bar_worker, args=(message_queue,), daemon=True).start()

    root.mainloop()
