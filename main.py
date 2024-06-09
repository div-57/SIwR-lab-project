import os
import sys
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import time

class Tracker:
    def __init__(self):
        self.tracks = {}
        self.track_id_counter = 0
        self.max_age = 1
        self.last_detections = []
        self.track_outputs = []

    def init_track(self, bbox):
        # Zdefiniowanie, że bieżące położenie obiektu zależy od poprzedniego położenia
        model = BayesianNetwork([('prev_x', 'x'), ('prev_y', 'y')])

        # Definicja rozkładu prawdopodobieństwa dla poprzednich stanów
        cpd_prev_x = TabularCPD(variable='prev_x', variable_card=2, values=[[0.5], [0.5]])
        cpd_prev_y = TabularCPD(variable='prev_y', variable_card=2, values=[[0.5], [0.5]])

        # Definicja rozkładu dla bieżących stanów
        cpd_x = TabularCPD(variable='x', variable_card=2,
                           values=[[0.7, 0.3], [0.3, 0.7]],
                           evidence=['prev_x'], evidence_card=[2])
        cpd_y = TabularCPD(variable='y', variable_card=2,
                           values=[[0.7, 0.3], [0.3, 0.7]],
                           evidence=['prev_y'], evidence_card=[2])

        # Dodanie prawdopodobieństwa do modelu
        model.add_cpds(cpd_prev_x, cpd_prev_y, cpd_x, cpd_y)

        # Tworzenie nowej ścieżki
        self.tracks[self.track_id_counter] = {'model': model, 'bbox': bbox, 'age': 0, 'total_visible_count': 1,
                                              'consecutive_invisible_count': 0}
        self.track_id_counter += 1

    # Aktualizacja ścieżki oraz liczników widoczności
    def update_tracks(self, detections):
        # Sprawdzenie, czy są detekcje
        if len(detections) == 0:
            for track in self.tracks.values():
                track['consecutive_invisible_count'] += 1
            self.delete_old_tracks()
            self.print_tracks([], detections)
            return

        # Macierz kosztów
        track_detections = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks.values()):
            for j, detection in enumerate(detections):
                track_detections[i, j] = 1 - self.calculate_iou(track['bbox'], detection)

        # Znalezienie optymalnego przypisania ścieżek do detekcji, minimalizując sumę kosztów w macierzy kosztów
        # row_ind, col_ind to indeksy wierszy i kolumn dla najlepszych przypisań
        row_ind, col_ind = linear_sum_assignment(track_detections)

        # Aktualizacja ścieżek
        updated_tracks = set()
        track_map = [-1] * len(detections)
        for i, track_id in enumerate(list(self.tracks.keys())):
            if i in row_ind:
                j = np.where(row_ind == i)[0][0]
                detection_idx = col_ind[j]

                self.tracks[track_id]['bbox'] = detections[detection_idx]

                self.tracks[track_id]['age'] += 1
                self.tracks[track_id]['total_visible_count'] += 1
                self.tracks[track_id]['consecutive_invisible_count'] = 0
                updated_tracks.add(track_id)
                track_map[detection_idx] = i

        # Inicjalizacja nowych ścieżek dla nieprzypisanych detekcji
        # Znalezienie indeksów detekcji, które nie zostały przypisane do żadnej ścieżki
        unmatched_detections = np.setdiff1d(np.arange(len(detections)), col_ind)
        for idx in unmatched_detections:
            self.init_track(detections[idx])

        # Aktualizacja dla liczników niewodoczności dla nieaktualizowanych ścieżek
        for track_id, track in self.tracks.items():
            if track_id not in updated_tracks:
                track['consecutive_invisible_count'] += 1

        # Usuwanie starych ścieżek i pokazywanie wyników
        self.delete_old_tracks()
        self.print_tracks(track_map, detections)
        self.last_detections = detections

    # Usuwanie ścieżek, które są za długo na obrazie
    def delete_old_tracks(self):
        track_ids_to_delete = []
        for track_id, track in self.tracks.items():
            if track['consecutive_invisible_count'] > self.max_age:
                track_ids_to_delete.append(track_id)
        for track_id in track_ids_to_delete:
            del self.tracks[track_id]

    # Intersection over Union - obliczenie współczynnika nakładania się dwóch prostokątów
    def calculate_iou(self, bbox1, bbox2):
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)

        # Powierzchnia przecięcia
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2

        # Powierzchnia sumu obu ramek ograniczających
        union_area = box1_area + box2_area - inter_area

        # Obliczony współczynnik
        return inter_area / union_area

    # Przetwarzanie i zapisanie wyników mapowania ścieżek do detekcji
    def print_tracks(self, track_map, detections):
        output = ' '.join(map(str, track_map))
        self.track_outputs.append(output)

def load_bboxes(bboxes_file):
    bboxes = {}
    with open(bboxes_file, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            filename = lines[i].strip()
            num_boxes = int(lines[i + 1].strip())
            boxes = []
            for j in range(num_boxes):
                box_data = lines[i + 2 + j].strip().split()
                x, y, w, h = map(float, box_data)
                boxes.append((x, y, w, h))
            bboxes[filename] = boxes
            i += num_boxes + 2
    return bboxes

# Wizualizacja wykrytych osób na obrazie (opcjonalne)
def visualize_tracks(frame_file, detections, tracked_bboxes):
    frame = cv2.imread(frame_file)

    for bbox in detections:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for bbox in tracked_bboxes:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Tracking', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Śledzenie pieszych
def track_pedestrians(dataset_path):
    frames_dir = os.path.join(dataset_path, "frames")
    bboxes_file = os.path.join(dataset_path, "bboxes.txt")
    tracker = Tracker()
    bboxes = load_bboxes(bboxes_file)
    frame_files = sorted(bboxes.keys())

    start_time = time.time()
    frame_times = []

    # Aktualizacja śledzenia w każdej ramce
    for frame_file in frame_files:
        frame_start_time = time.time()
        frame_path = os.path.join(frames_dir, frame_file)
        detections = bboxes[frame_file]
        tracker.update_tracks(detections)
        tracked_bboxes = [track['bbox'] for track in tracker.tracks.values()]
        # Wizualizacja
        # visualize_tracks(frame_path, detections, tracked_bboxes)
        frame_end_time = time.time()
        frame_times.append(frame_end_time - frame_start_time)

    end_time = time.time()
    total_time = end_time - start_time
    average_frame_time = sum(frame_times) / len(frame_times) if frame_times else 0

    # Obliczanie czasu wykonania
    print(f"Czas wykonania całego kodu: {total_time:.2f} sekundy")
    print(f"Średni czas na klatkę: {average_frame_time:.2f} sekundy")

    return tracker.track_outputs

# Zliczanie znaków w danym wierszu
def digit_counts(lines):
    digit_counts = []
    for line in lines:
        digit_count = sum(char.isdigit() for char in line)
        digit_counts.append(digit_count)
    return digit_counts

# Wczytanie pliku jako wektor
def load_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        vector = []
        for line in lines:
            elements = line.strip().split()
            vector.extend([int(el) for el in elements if el.isdigit()])
    return vector

# Sortowanie i porównywanie wektorów
def sort_and_compare_tracks(track_outputs, gt_vector):
    sorted_tracks = [sorted(map(int, line.split())) for line in track_outputs]
    gt_vector_grouped = []
    index = 0

    for line in track_outputs:
        length = len(line.split())
        gt_vector_grouped.append(gt_vector[index:index + length])
        index += length

    sorted_gt_vector = [sorted(line) for line in gt_vector_grouped]
    comparisons = []

    for i in range(min(len(sorted_tracks), len(sorted_gt_vector))):
        track_line = sorted_tracks[i]
        gt_line = sorted_gt_vector[i]
        comparison = [track == gt for track, gt in zip(track_line, gt_line)]
        comparisons.append(comparison)

    return comparisons

accuracy_bboxes = 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_dataset>")
        sys.exit(1)

    dataset_path = sys.argv[1]

    # Liczenie dokładności dla bboxes.txt i dla bboxes_gt.txt
    track_outputs = track_pedestrians(dataset_path)
    digits = digit_counts(track_outputs)

    input_file = 'bboxes_filtered.txt'
    bboxes_vector = load_file(input_file)

    for i in range(len(bboxes_vector)):
        if digits[i] == bboxes_vector[i]:
            accuracy_bboxes += 1

    accuracy_bboxes_percent = accuracy_bboxes / len(bboxes_vector) * 100
    print("Dokładność - bboxes.txt: ")
    print(f"{accuracy_bboxes_percent:.2f}%")

    gt_vector_file = 'bboxes_gt_vector_filtered.txt'
    gt_vector = load_file(gt_vector_file)

    comparisons = sort_and_compare_tracks(track_outputs, gt_vector)

    accuracy_bboxes_gt = sum(all(comp) for comp in comparisons)
    accuracy_bboxes_gt_percent = accuracy_bboxes_gt / len(comparisons) * 100 if comparisons else 0

    print("Dokładność - bboxes_gt.txt: ")
    print(f"{accuracy_bboxes_gt_percent:.2f}%")