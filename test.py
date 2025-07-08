import torch
import tkinter as tk
from tkinter import filedialog, Label, ttk
from PIL import Image, ImageDraw
import numpy as np
from convautoencodermodel import ConvAutoencoder64, ConvAutoencoder128, ConvAutoencoder256
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os

img_path = None
model64 = None
model128 = None
model256 = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cache = {}
stable_anomalies = []
scale_test_completed = False

def select_image():
    global img_path
    file_path = filedialog.askopenfilename(title="Виберіть зображення",
                                           filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        img_path = file_path
        status_label.config(text=f"Обране зображення: {os.path.basename(file_path)}")

def load_model(size):
    global model64, model128, model256
    
    if size == 64 and model64 is None:
        model64 = ConvAutoencoder64()
        checkpoint = torch.load("convautoencoders.pth", map_location=device)
        model64.load_state_dict(checkpoint['model64'])
        model64.to(device)
        model64.eval()
        return model64
    elif size == 128 and model128 is None:
        model128 = ConvAutoencoder128()
        checkpoint = torch.load("convautoencoders.pth", map_location=device)
        model128.load_state_dict(checkpoint['model128'])
        model128.to(device)
        model128.eval()
        return model128
    elif size == 256 and model256 is None:
        model256 = ConvAutoencoder256()
        checkpoint = torch.load("convautoencoders.pth", map_location=device)
        model256.load_state_dict(checkpoint['model256'])
        model256.to(device)
        model256.eval()
        return model256
    else:
        if size == 64:
            return model64
        elif size == 128:
            return model128
        elif size == 256:
            return model256

def split_image(image_path, patch_size, overlap_percent):
    cache_key = f"{image_path}_{patch_size}_{overlap_percent}"
    if cache_key in cache:
        return cache[cache_key]
    
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    target_w = (w // patch_size) * patch_size
    target_h = (h // patch_size) * patch_size

    if w != target_w or h != target_h:
        image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    patches = []
    overlap = int(patch_size * (overlap_percent / 100.0))
    step_size = patch_size - overlap

    if step_size <= 0:
        step_size = patch_size

    for i in range(0, target_w - patch_size + 1, step_size):
        for j in range(0, target_h - patch_size + 1, step_size):
            patch = image.crop((i, j, min(i + patch_size, target_w), min(j + patch_size, target_h)))
            patches.append({
                'patch': patch,
                'row': j // step_size,
                'col': i // step_size
            })
    
    result = (image, patches)
    cache[cache_key] = result
    return result

def process_patch(patch_info, model, device):
    patch = patch_info['patch']
    row = patch_info['row']
    col = patch_info['col']
    
    tensor_patch = torch.tensor(np.array(patch).transpose(2, 0, 1), 
                              dtype=torch.float32).unsqueeze(0).to(device) / 255.0
    tensor_patch = tensor_patch * 2.0 - 1.0
    
    with torch.no_grad():
        encoded_vector = model.encode(tensor_patch)
    
    vector_np = encoded_vector.squeeze().cpu().numpy().flatten()
    
    return {
        'row': row,
        'col': col,
        'vector': vector_np
    }

def process_and_decode_patches(patches_with_indices, model, device, batch_size=32):
    encoded_vectors_with_indices = []
    status_label.config(text=f"Обробка {len(patches_with_indices)} патчів батчами...")
    window.update()

    tensors = []
    row_col_info = []
    for item in patches_with_indices:
        np_patch = np.array(item['patch']).transpose(2, 0, 1) / 255.0
        np_patch = np_patch * 2.0 - 1.0
        tensor = torch.tensor(np_patch, dtype=torch.float32)
        tensors.append(tensor)
        row_col_info.append({'row': item['row'], 'col': item['col']})

    all_tensors = torch.stack(tensors).to(device)

    with torch.no_grad():
        for i in range(0, len(all_tensors), batch_size):
            batch = all_tensors[i:i+batch_size]
            encoded_batch = model.encode(batch)

            for j in range(encoded_batch.size(0)):
                vector_np = encoded_batch[j].cpu().numpy().flatten()
                encoded_vectors_with_indices.append({
                    'row': row_col_info[i + j]['row'],
                    'col': row_col_info[i + j]['col'],
                    'vector': vector_np
                })

    max_len = max(len(item['vector']) for item in encoded_vectors_with_indices)
    for item in encoded_vectors_with_indices:
        if len(item['vector']) < max_len:
            item['vector'] = np.pad(item['vector'], (0, max_len - len(item['vector'])), 'constant')

    return encoded_vectors_with_indices

def get_lof_params(patch_size, custom_contamination=None):
    if custom_contamination is not None:
        contamination = custom_contamination
    else:
        if patch_size <= 64:
            contamination = 0.12
        elif patch_size <= 128:
            contamination = 0.15
        else:
            contamination = 0.17

    if patch_size <= 64:
        return 30, contamination
    elif patch_size <= 128:
        return 45, contamination
    else:
        return 35, contamination

def test_scale_robustness(grid_size, percent, anomaly_percent):
    global scale_test_completed
    
    if not img_path:
        status_label.config(text="Помилка: спочатку оберіть зображення")
        return
    
    try:
        patch_size = int(grid_size.split('x')[0])
        percent_overlap = float(percent)
        
        if anomaly_percent and anomaly_percent.strip():
            custom_contamination = float(anomaly_percent.strip()) / 100.0
        else:
            custom_contamination = None
        
        start_time = time.time()
        current_model = load_model(patch_size)
        
        original_img = Image.open(img_path).convert("RGB")
        w, h = original_img.size
        
        smaller_img = original_img.resize((w//2, h//2), Image.Resampling.LANCZOS)
        larger_img = original_img.resize((w*2, h*2), Image.Resampling.LANCZOS)
        
        images = [original_img, smaller_img, larger_img]
        image_names = ["оригінал", "зменшене", "збільшене"]
        all_centers = []
        
        status_label.config(text="Обробка зображень різних масштабів...")
        window.update()
        
        for i, img in enumerate(images):
            temp_path = f"temp_scale_{i}.png"
            img.save(temp_path)
            
            _, patches_with_indices = split_image(temp_path, patch_size, percent_overlap)

            encoded_vectors_with_indices = process_and_decode_patches(
                patches_with_indices, current_model, device
            )
            
            encoded_vectors_np = np.array([item['vector'] for item in encoded_vectors_with_indices])
            row_col_indices = [{'row': item['row'], 'col': item['col']} for item in encoded_vectors_with_indices]

            n_components = 5
            pca = PCA(n_components=n_components)
            reduced_vectors = pca.fit_transform(encoded_vectors_np)
            
            n_neighbors, contamination = get_lof_params(patch_size, custom_contamination)
            anomalies_indices = detect_anomalies_with_lof(
                reduced_vectors, n_neighbors=n_neighbors, contamination=contamination
            )
            
            centers = []
            overlap = int(patch_size * (percent_overlap / 100.0))
            step_size = patch_size - overlap
            if step_size <= 0:
                step_size = patch_size
                
            for idx in anomalies_indices:
                row = row_col_indices[idx]['row']
                col = row_col_indices[idx]['col']
                center_x = col * step_size + patch_size // 2
                center_y = row * step_size + patch_size // 2
                centers.append((center_x, center_y))
            
            all_centers.append(centers)
            
            os.remove(temp_path)
        
        diff_original_smaller = []
        diff_original_larger = []
        
        for center_orig in all_centers[0]:
            if all_centers[1]:
                min_dist = float('inf')
                for center_small in all_centers[1]:
                    scaled_center = (center_small[0] * 2, center_small[1] * 2)
                    dist = np.sqrt((center_orig[0] - scaled_center[0])**2 + (center_orig[1] - scaled_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                diff_original_smaller.append(min_dist)
            
            if all_centers[2]:
                min_dist = float('inf')
                for center_large in all_centers[2]:
                    scaled_center = (center_large[0] / 2, center_large[1] / 2)
                    dist = np.sqrt((center_orig[0] - scaled_center[0])**2 + (center_orig[1] - scaled_center[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                diff_original_larger.append(min_dist)
        
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if diff_original_smaller:
            ax1.hist(diff_original_smaller, bins=20, alpha=0.7, color='blue', edgecolor='black', density=True)
            ax1.set_title('Розподіл відстаней: оригінал vs зменшене')
            ax1.set_xlabel('Відстань (пікселі)')
            ax1.set_ylabel('Частота')
            ax1.grid(True, alpha=0.3)
        
        if diff_original_larger:
            ax2.hist(diff_original_larger, bins=20, alpha=0.7, color='red', edgecolor='black', density=True)
            ax2.set_title('Розподіл відстаней: оригінал vs збільшене')
            ax2.set_xlabel('Відстань (пікселі)')
            ax2.set_ylabel('Частота')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=result_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        end_time = time.time()
        scale_test_completed = True
        
        status_label.config(
            text=f"Тест масштабування завершено за {end_time - start_time:.2f} сек. "
                 f"Аномалій: {len(all_centers[0])}/{len(all_centers[1])}/{len(all_centers[2])}"
        )
        
    except Exception as e:
        status_label.config(text=f"Помилка: {str(e)}")
        print(f"Помилка: {e}")

def detect_anomalies_with_lof(vectors, n_neighbors=30, contamination=0.1):
    scaler = StandardScaler()
    scaled_vectors = scaler.fit_transform(vectors)
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    anomaly_labels = lof.fit_predict(scaled_vectors)
    
    anomaly_indices = np.where(anomaly_labels == -1)[0]
    
    return anomaly_indices

def visualize_anomalies_on_image(image, row_col_indices, anomalies_indices, patch_size, overlap_percent):
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    width, height = image.size
    
    overlap = int(patch_size * (overlap_percent / 100.0))
    step_size = patch_size - overlap
    if step_size <= 0:
        step_size = patch_size
    
    for index in anomalies_indices:
        row = row_col_indices[index]['row']
        col = row_col_indices[index]['col']
        x0, y0 = col * step_size, row * step_size
        x1, y1 = min(x0 + patch_size, width), min(y0 + patch_size, height)

        draw.rectangle([(x0, y0), (x1, y1)], outline='red', width=18)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img_copy)
    ax.set_title(f'Виявлені аномалії (патч {patch_size}x{patch_size})')
    ax.axis('off')

    for widget in result_frame.winfo_children():
        widget.destroy()
    
    canvas = FigureCanvasTkAgg(fig, master=result_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()
    
    status_label.config(
        text=f"Знайдено {len(anomalies_indices)} аномалій з {len(row_col_indices)} патчів."
    )

def visualize_anomalies_3d(reduced_vectors, anomalies_indices):
    for widget in result_frame.winfo_children():
        widget.destroy()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    normal_indices = np.setdiff1d(np.arange(len(reduced_vectors)), anomalies_indices)
    observations = reduced_vectors[normal_indices]
    anomalies = reduced_vectors[anomalies_indices]
    
    ax.scatter(observations[:, 0], observations[:, 1], observations[:, 2], 
               c='blue', marker='o', label='Нормальні')
    ax.scatter(anomalies[:, 0], anomalies[:, 1], anomalies[:, 2], 
               c='red', marker='x', s=100, label='Аномалії')
    
    ax.set_xlabel('ГК 1')
    ax.set_ylabel('ГК 2')
    ax.set_zlabel('ГК 3')
    ax.set_title('3D візуалізація аномалій')
    ax.legend()
    
    canvas = FigureCanvasTkAgg(fig, master=result_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()
    
    status_label.config(
        text=f"Знайдено {len(anomalies_indices)} аномалій з {len(reduced_vectors)} точок."
    )

def detect_anomalies(grid_size, percent, anomaly_percent):
    if not img_path:
        status_label.config(text="Помилка: спочатку оберіть зображення")
        return

    if anomaly_percent and anomaly_percent.strip():
        custom_contamination = float(anomaly_percent.strip()) / 100.0
    else:
        custom_contamination = None
    
    try:
        patch_size = int(grid_size.split('x')[0])
        percent_overlap = float(percent)
        
        start_time = time.time()
        
        current_model = load_model(patch_size)
        
        original_img, patches_with_indices = split_image(img_path, patch_size, percent_overlap)
        
        encoded_vectors_with_indices = process_and_decode_patches(
            patches_with_indices, current_model, device
        )
        
        encoded_vectors_np = np.array([item['vector'] for item in encoded_vectors_with_indices])
        row_col_indices = [{'row': item['row'], 'col': item['col']} for item in encoded_vectors_with_indices]
        
        n_components = 5
        pca = PCA(n_components=n_components)
        reduced_vectors = pca.fit_transform(encoded_vectors_np)
        
        n_neighbors, contamination = get_lof_params(patch_size, custom_contamination)
        
        anomalies_indices = detect_anomalies_with_lof(
            reduced_vectors, 
            n_neighbors=n_neighbors, 
            contamination=contamination
        )
        
        visualize_anomalies_on_image(
            original_img, 
            row_col_indices, 
            anomalies_indices, 
            patch_size, 
            percent_overlap
        )
        
        end_time = time.time()
        
        status_label.config(
            text=f"Аналіз виконано за {end_time - start_time:.2f} сек. Знайдено {len(anomalies_indices)} аномалій."
        )
        
    except Exception as e:
        status_label.config(text=f"Помилка: {str(e)}")
        print(f"Помилка: {e}")
        
def filter_stable_anomalies(grid_size, percent, anomaly_percent,
                            quantile_threshold=50):
    global stable_anomalies

    if not img_path:
        status_label.config(text="Помилка: спочатку оберіть зображення")
        return

    try:
        patch_size       = int(grid_size.split('x')[0])
        percent_overlap  = float(percent)
        q_percentile     = float(quantile_threshold)
        custom_contam    = (float(anomaly_percent)/100.0
                            if anomaly_percent.strip() else None)

        start = time.time()
        model = load_model(patch_size)

        n_neighbors, contamination = get_lof_params(patch_size,
                                                    custom_contam)
        overlap   = int(patch_size * percent_overlap / 100.0)
        step_size = patch_size - overlap or patch_size

        def centers_at_scale(img_or_path, scale):
            if isinstance(img_or_path, str):
                _, patches = split_image(img_or_path,
                                         patch_size, percent_overlap)
            else:
                tmp = f"_tmp_{scale}.png"
                img_or_path.save(tmp)
                _, patches = split_image(tmp, patch_size, percent_overlap)
                os.remove(tmp)

            encoded = process_and_decode_patches(patches, model, device)
            vecs    = np.array([e['vector'] for e in encoded])
            if vecs.size == 0:
                return []

            pca = PCA(n_components=5).fit_transform(vecs)
            idxs = detect_anomalies_with_lof(pca,
                                             n_neighbors, contamination)

            result = []
            for idx in idxs:
                row = encoded[idx]['row']
                col = encoded[idx]['col']
                cx  = col * step_size + patch_size // 2
                cy  = row * step_size + patch_size // 2
                result.append((cx, cy, idx))
            return result

        orig_centers  = centers_at_scale(img_path, 1.0)

        img_orig      = Image.open(img_path).convert("RGB")
        w, h          = img_orig.size
        img_small     = img_orig.resize((w//2, h//2), Image.Resampling.LANCZOS)
        img_large     = img_orig.resize((w*2, h*2), Image.Resampling.LANCZOS)

        small_centers = centers_at_scale(img_small, 0.5)
        large_centers = centers_at_scale(img_large, 2.0)

        small_scaled  = [(2*x,   2*y)   for (x, y, _) in small_centers]
        large_scaled  = [(0.5*x, 0.5*y) for (x, y, _) in large_centers]

        delta_small, delta_large = [], []
        for ox, oy, _ in orig_centers:
            d1 = (min(np.hypot(ox - sx, oy - sy)
                      for sx, sy in small_scaled)
                  if small_scaled else np.inf)
            d2 = (min(np.hypot(ox - lx, oy - ly)
                      for lx, ly in large_scaled)
                  if large_scaled else np.inf)
            delta_small.append(d1)
            delta_large.append(d2)

        tau_small = np.percentile(delta_small, q_percentile)
        tau_large = np.percentile(delta_large, q_percentile)

        stable_idx = [orig_centers[i][2]
                      for i in range(len(orig_centers))
                      if delta_small[i] <= tau_small
                      and delta_large[i] <= tau_large]

        _, orig_patches = split_image(img_path, patch_size, percent_overlap)
        encoded_orig = process_and_decode_patches(orig_patches, model, device)
        rc_indices   = [{'row': e['row'], 'col': e['col']} for e in encoded_orig]

        visualize_anomalies_on_image(img_orig, rc_indices,
                                     stable_idx, patch_size, percent_overlap)

        stable_anomalies[:] = stable_idx
        elapsed = time.time() - start
        status_label.config(
            text=(f"Фільтрація завершена за {elapsed:.2f} c; "
                  f"залишено {len(stable_idx)} / {len(orig_centers)} аномалій "
                  f"(квантиль {q_percentile}%). "
                  f"Пороги: small={tau_small:.1f}px, large={tau_large:.1f}px"))
    except Exception as e:
        status_label.config(text=f"Помилка: {e}")
        print("Помилка:", e)

def detect_anomalies_3d(grid_size, percent, anomaly_percent):
    if not img_path:
        status_label.config(text="Помилка: спочатку оберіть зображення")
        return
    
    if anomaly_percent and anomaly_percent.strip():
        custom_contamination = float(anomaly_percent.strip()) / 100.0
    else:
        custom_contamination = None
    
    try:
        patch_size = int(grid_size.split('x')[0])
        percent_overlap = float(percent)
        
        start_time = time.time()

        current_model = load_model(patch_size)

        original_img, patches_with_indices = split_image(img_path, patch_size, percent_overlap)
        
        encoded_vectors_with_indices = process_and_decode_patches(
            patches_with_indices, current_model, device
        )
        
        encoded_vectors_np = np.array([item['vector'] for item in encoded_vectors_with_indices])
        
        pca = PCA(n_components=5)
        reduced_vectors = pca.fit_transform(encoded_vectors_np)
        
        n_neighbors, contamination = get_lof_params(patch_size, custom_contamination)
        
        anomalies_indices = detect_anomalies_with_lof(
            reduced_vectors, 
            n_neighbors=n_neighbors, 
            contamination=contamination
        )
        
        visualize_anomalies_3d(reduced_vectors, anomalies_indices)
        
        end_time = time.time()
        
        status_label.config(
            text=f"3D аналіз виконано за {end_time - start_time:.2f} сек. Знайдено {len(anomalies_indices)} аномалій."
        )
        
    except Exception as e:
        status_label.config(text=f"Помилка: {str(e)}")
        print(f"Помилка: {e}")
        
def plot_parallel_coordinates(grid_size, percent, anomaly_percent):
    if not img_path:
        status_label.config(text="Помилка: спочатку оберіть зображення")
        return
    
    try:
        patch_size = int(grid_size.split('x')[0])
        percent_overlap = float(percent)
        
        start_time = time.time()
        current_model = load_model(patch_size)
        
        original_img, patches_with_indices = split_image(img_path, patch_size, percent_overlap)
        
        encoded_vectors_with_indices = process_and_decode_patches(
            patches_with_indices, current_model, device
        )
        
        encoded_vectors_np = np.array([item['vector'] for item in encoded_vectors_with_indices])
        
        n_components = 10
        pca = PCA(n_components=n_components)
        reduced_vectors = pca.fit_transform(encoded_vectors_np)
        
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized_vectors = scaler.fit_transform(reduced_vectors)
        
        for widget in result_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for i in range(len(normalized_vectors)):
            ax.plot(range(n_components), normalized_vectors[i], 'b-', alpha=0.3, linewidth=0.5)
        
        ax.set_xlabel('Ознаки (Головні компоненти)')
        ax.set_ylabel('Нормалізовані значення')
        ax.set_title('Паралельні координати')
        ax.set_xticks(range(n_components))
        ax.grid(True, alpha=0.3)
        
        variability = pca.explained_variance_ratio_
        
        labels = []
        for i in range(n_components):
            labels.append(f'ГК{i+1}\n{variability[i]:.3f}')
        
        ax.set_xticklabels(labels)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=result_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        canvas.draw()
        
        end_time = time.time()
        
        status_label.config(
            text=f"Паралельні координати побудовано за {end_time - start_time:.2f} сек. "
                 f"Загальна варіація: {sum(variability):.3f}"
        )
        
    except Exception as e:
        status_label.config(text=f"Помилка: {str(e)}")
        print(f"Помилка: {e}")


window = tk.Tk()
window.title("Виявлення аномалій на зображеннях")
window.geometry("1000x800")

control_frame = tk.Frame(window)
control_frame.pack(pady=10)

select_button = tk.Button(control_frame, text="Вибрати зображення", command=select_image)
select_button.grid(row=0, column=0, padx=10)

options_frame = tk.Frame(window)
options_frame.pack(pady=5)

grid_label = tk.Label(options_frame, text="Розмір сітки:")
grid_label.grid(row=0, column=0, padx=5)

grid_var = tk.StringVar(window)
grid_var.set("64x64")
grid_options = ["64x64", "128x128", "256x256"]
grid_menu = ttk.OptionMenu(options_frame, grid_var, grid_var.get(), *grid_options)
grid_menu.grid(row=0, column=1, padx=5)

overlap_label = tk.Label(options_frame, text="Перекриття:")
overlap_label.grid(row=0, column=2, padx=5)

overlap_var = tk.StringVar(window)
overlap_var.set("0")
overlap_options = ["0", "25", "50", "75"]
overlap_menu = ttk.OptionMenu(options_frame, overlap_var, overlap_var.get(), *overlap_options)
overlap_menu.grid(row=0, column=3, padx=5)

anomaly_percent_label = tk.Label(options_frame, text="% аномалій:")
anomaly_percent_label.grid(row=0, column=4, padx=5)

anomaly_percent_var = tk.StringVar(window)
anomaly_percent_var.set("")
anomaly_percent_entry = tk.Entry(options_frame, textvariable=anomaly_percent_var, width=8)
anomaly_percent_entry.grid(row=0, column=5, padx=5)

buttons_frame = tk.Frame(window)
buttons_frame.pack(pady=10)

run_button = tk.Button(
    buttons_frame, 
    text="Пошук аномалій на зображенні", 
    command=lambda: detect_anomalies(grid_var.get(), overlap_var.get(),anomaly_percent_var.get())
)
run_button.grid(row=0, column=0, padx=10)

run_button3d = tk.Button(
    buttons_frame, 
    text="3D візуалізація аномалій", 
    command=lambda: detect_anomalies_3d(grid_var.get(), overlap_var.get(),anomaly_percent_var.get())
)
run_button3d.grid(row=0, column=1, padx=10)

run_button_scale = tk.Button(
    buttons_frame, 
    text="Тест стійкості масштабування", 
    command=lambda: test_scale_robustness(grid_var.get(), overlap_var.get(),anomaly_percent_var.get())
)
run_button_scale.grid(row=0, column=2, padx=10)

run_button_filter = tk.Button(
    buttons_frame, 
    text="Фільтрація стійких аномалій", 
    command=lambda: filter_stable_anomalies(grid_var.get(), overlap_var.get(), anomaly_percent_var.get())
)
run_button_filter.grid(row=1, column=0, padx=10, pady=5)

status_label = tk.Label(window, text="Оберіть зображення для аналізу", font=("Helvetica", 10))
status_label.pack(pady=5)

result_frame = tk.Frame(window)
result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

run_button_parallel = tk.Button(
    buttons_frame, 
    text="Паралельні координати", 
    command=lambda: plot_parallel_coordinates(grid_var.get(), overlap_var.get(), anomaly_percent_var.get())
)
run_button_parallel.grid(row=1, column=1, padx=10, pady=5)

def select_image():
    global img_path
    file_path = filedialog.askopenfilename(
        title="Виберіть зображення",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")] 
    )
    if file_path:
        img_path = file_path
        status_label.config(text=f"Обране зображення: {os.path.basename(file_path)}")

window.mainloop()