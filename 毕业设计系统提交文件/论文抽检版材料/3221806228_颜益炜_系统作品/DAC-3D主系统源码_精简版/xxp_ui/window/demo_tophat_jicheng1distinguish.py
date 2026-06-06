import cv2
import numpy as np
import os
import gc




def stack_images(scale, img_array):
    """
    Stacks images horizontally and vertically.
    scale: percentage to scale the images
    img_array: a list of lists of images. Each inner list is a row.
               Example: [[img1, img2], [img3, img4]]
    """
    rows = len(img_array)
    if not rows: return None
    if not img_array[0]: return None
    cols = len(img_array[0])
    if not cols: return None

    first_img_for_size = None
    for r in img_array:
        if r and r[0] is not None and hasattr(r[0], 'shape'):
            first_img_for_size = r[0]
            break
    if first_img_for_size is None:
        print("Error: No valid base image found for stacking.")
        return None

    base_height, base_width = first_img_for_size.shape[:2]
    target_width = int(base_width * scale)
    target_height = int(base_height * scale)

    processed_rows = []
    for r_idx in range(rows):
        processed_cols = []
        for c_idx in range(cols):
            img = img_array[r_idx][c_idx]
            if img is None:
                img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                cv2.putText(img, "N/A", (5, target_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
            else:
                img = cv2.resize(img, (target_width, target_height),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            processed_cols.append(img)
        if processed_cols:
            processed_rows.append(np.hstack(processed_cols))
    if not processed_rows: return None
    ver = np.vstack(processed_rows)
    return ver


def add_text_to_image(image, text, position=(10, 20), font_scale=0.5, color=(255, 255, 0), thickness=1):
    img_to_draw_on = image
    if image is None:
        print(f"Warning: add_text_to_image received a None image for text: {text}")
        dummy_height = 50;
        dummy_width = 100
        if position[1] > dummy_height: dummy_height = position[1] + 10
        img_to_draw_on = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8)
    elif len(image.shape) == 2:
        img_to_draw_on = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 1:
        img_to_draw_on = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_to_draw_on, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    return img_to_draw_on


def process_image_opencv(input_numpy_image, is_color=False):
    """
    Processes an input image (NumPy array) to detect and classify defects.
    Only displays the final classified image.

    Args:
        input_numpy_image (np.ndarray): The input image as a NumPy array.
                                        Should be grayscale (2D) or BGR (3D).
        is_color (bool): True if input_numpy_image is a BGR color image,
                         False if it's already grayscale.
    Returns:
        np.ndarray: The final classified image (resized), or None if processing fails.
    """
    if input_numpy_image is None:
        print("Error: Input NumPy image is None.")
        return None

    print("Processing NumPy image...")

    if is_color:
        if len(input_numpy_image.shape) == 3 and input_numpy_image.shape[2] == 3:
            gray_image_orig = cv2.cvtColor(input_numpy_image, cv2.COLOR_BGR2GRAY)
        else:
            print("Error: 'is_color' is True, but input image is not a 3-channel BGR image.")
            return None
    else:
        if len(input_numpy_image.shape) == 2:
            gray_image_orig = input_numpy_image.copy()
        elif len(input_numpy_image.shape) == 3 and input_numpy_image.shape[2] == 1:
            gray_image_orig = input_numpy_image[:, :, 0].copy()
        else:
            print("Error: 'is_color' is False, but input image is not a 2D grayscale image or single channel.")
            return None

    if gray_image_orig is None:
        print("Error: Could not obtain grayscale image from input.")
        return None

    print("Grayscale image obtained successfully!")
    gray_image = gray_image_orig.copy()

    display_image_classification = cv2.cvtColor(gray_image_orig, cv2.COLOR_GRAY2BGR)
    original_height, original_width = gray_image.shape[:2]

    # 1. 高斯滤波
    gauss_image = cv2.GaussianBlur(gray_image, (19, 19), 3)

    # 2. 顶帽变换
    se_tophat_diameter = 100
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_tophat_diameter, se_tophat_diameter))
    image_tophat = cv2.morphologyEx(gauss_image, cv2.MORPH_TOPHAT, kernel_tophat)
    del gauss_image

    # 3. Sobel边缘检测 (使用CV_32F)
    grad_x = cv2.Sobel(image_tophat, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_tophat, cv2.CV_32F, 0, 1, ksize=3)
    edge_amplitude_float = np.abs(grad_x) + np.abs(grad_y)
    del grad_x, grad_y
    del image_tophat
    gc.collect()

    # 4. 边缘振幅图像缩放
    alpha_scale = 5
    image_scaled = cv2.convertScaleAbs(edge_amplitude_float, alpha=alpha_scale, beta=0)
    del edge_amplitude_float
    gc.collect()

    # 5. Otsu二值化
    used_threshold, region1_binary = cv2.threshold(image_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    del image_scaled
    gc.collect()


    # 6. 孔洞填充
    region_fill_up = region1_binary.copy()
    del region1_binary
    gc.collect()
    contours_for_fill, hierarchy_for_fill = cv2.findContours(region_fill_up, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy_for_fill is not None:
        for i_contour, contour_to_fill in enumerate(contours_for_fill):
            if hierarchy_for_fill[0][i_contour][3] != -1:
                cv2.drawContours(region_fill_up, [contour_to_fill], 0, 255, -1)

    # 7. 开运算
    opening_kernel_size_val = 3
    opening_kernel_size = (opening_kernel_size_val, opening_kernel_size_val)
    kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, opening_kernel_size)
    region_opened = cv2.morphologyEx(region_fill_up, cv2.MORPH_OPEN, kernel_opening)
    del region_fill_up
    gc.collect()

    # 8. 连通组件分析
    num_labels, connected_regions_labels, stats, centroids = cv2.connectedComponentsWithStats(region_opened,
                                                                                              connectivity=8)
    # 9. 按面积筛选
    min_area = 10
    max_area = 40000
    gray_value_threshold_for_scratch = 160
    selected_regions_data = []

    for i_label in range(1, num_labels):
        area = stats[i_label, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            temp_mask = (connected_regions_labels == i_label).astype(np.uint8) * 255
            contours, _ = cv2.findContours(temp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            del temp_mask

            if contours:
                contour = contours[0]
                rect = cv2.minAreaRect(contour)
                center_rc_from_stats = (centroids[i_label][1], centroids[i_label][0])  # row, col
                selected_regions_data.append({
                    'rect': rect,
                    'center_rc': center_rc_from_stats,
                    'area': area,

                    'label_id': i_label
                })

    num_selected_after_area = len(selected_regions_data)

    # 10. 分类和绘制标记 (包含灰度值检查)
    final_defects_to_draw = []

    for data_item in selected_regions_data:
        rect = data_item['rect']
        (center_x_rect, center_y_rect), (rect_width, rect_height), angle_deg = rect

        if rect_width < 1 or rect_height < 1:
            aspect_ratio = 1
        else:
            aspect_ratio = max(rect_width, rect_height) / min(rect_width, rect_height)

        is_scratch_candidate = (aspect_ratio >= 4)

        should_draw = True
        mark_text = ""
        box_color = (0, 0, 0)

        if is_scratch_candidate:
            # 这是候选划痕，检查其灰度值
            component_mask_for_gray = (connected_regions_labels == data_item['label_id']).astype(np.uint8)

            if np.any(component_mask_for_gray):
                mean_gray_value = cv2.mean(gray_image_orig, mask=component_mask_for_gray)[0]

                if mean_gray_value > gray_value_threshold_for_scratch:
                    should_draw = False
                else:
                    mark_text = '1'
                    box_color = (0, 255, 0)  # Green for '1'
            else:
                should_draw = False
        else:
            mark_text = '0'
            box_color = (255, 0, 0)  # Blue for '0'

        if should_draw:
            final_defects_to_draw.append({
                'rect': rect,
                'center_rc': data_item['center_rc'],
                'mark_text': mark_text,
                'box_color': box_color
            })

    del connected_regions_labels, region_opened, stats, centroids
    gc.collect()

    for defect_info in final_defects_to_draw:
        rect = defect_info['rect']
        center_r, center_c = map(int, defect_info['center_rc'])
        mark_text = defect_info['mark_text']
        box_color = defect_info['box_color']

        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(display_image_classification, [box], 0, box_color, 2)

        text_pos_r = int(center_r + 20)
        text_pos_c = int(center_c)
        if 0 <= text_pos_r < original_height and 0 <= text_pos_c < original_width:
            cv2.putText(display_image_classification, mark_text, (text_pos_c, text_pos_r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)  # White text

    # 11. 调整最终输出图像大小并显示
    new_width = original_width // 2
    new_height = original_height // 2
    resized_classified_image = cv2.resize(display_image_classification, (new_width, new_height),
                                          interpolation=cv2.INTER_AREA)

    print("Processing complete. Displaying final classified image.")
    cv2.imshow("Defect Classification (0:Pit, 1:Scratch) - Resized Final", resized_classified_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return resized_classified_image


if __name__ == '__main__':
    img = cv2.imread(r'D:\zycgit\ZDevelop_Confocal\xxp_ui\testfocus.png',0)
    print(img.shape)
    dst = process_image_opencv(img)
