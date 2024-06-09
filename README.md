# Sztuczna Inteligencja w Robotyce - Laboratorium
# Raport z działania algorytmu śledzenia - Projekt

### 1. Ogólny zarys

Projekt implementuje algorytm śledzenia pieszych wykorzystujący sieci Bayesa oraz algorytm węgierski do asocjacji obiektów. Tracker jest zaprojektowany do identyfikacji i śledzenia wielu pieszych na kolejnych klatkach wideo.

### 2. Koncepcja i zasada działania

#### 2.1. Inicjalizacja Trackera

Klasa **Tracker** jest odpowiedzialna za utrzymanie i aktualizację śledzonych obiektów (pieszych). Każdy track reprezentuje unikalny obiekt zidentyfikowany na klatkach.
- Atrybuty:
  - **tracks**: Słownik przechowujący obecnie aktywne tracki.
  - **track_id_counter**: Licznik do przypisywania unikalnych ID dla każdego tracka.
  - **max_age**: Maksymalny wiek tracka przed jego usunięciem, jeśli nie zostanie zaktualizowany.
  - **last_detections**: Przechowuje ostatni zestaw detekcji.
  - **track_outputs**: Zapisuje wyniki procesu śledzenia dla każdej klatki.

#### 2.2. Inicjalizacja tracków
Tracki są inicjalizowane przy użyciu sieci Bayesa do modelowania przejść stanów pozycji obiektów.
- Struktura Sieci Bayesa:
  - Zmienne: **prev_x**, **prev_y** (poprzednie pozycje), **x**, **y** (aktualne pozycje).
  - CPD (Tablice Rozkładów Warunkowych): Są zdefiniowane do modelowania przejść stanów z określonymi prawdopodobieństwami.

#### 2.3 Aktualizacja tracków
Metoda **update_tracks** aktualizuje istniejące tracki za pomocą nowych detekcji oraz inicjalizuje nowe tracki dla niepasujących detekcji.
- Kroki:
  - Obliczanie współczynnika **IoU** (Intersection over Union) dla każdej detekcji względem istniejących tracków.
  - Użycie algorytmu węgierskiego do asocjacji detekcji z trackami na podstawie wyników IoU.
  - Aktualizacja tracków za pomocą skojarzonych detekcji.
  - Inicjalizacja nowych tracków dla detekcji, które nie mogły być skojarzone z żadnym istniejącym trackiem.
  - Zwiększenie **consecutive_invisible_count** dla tracków, które nie zostały zaktualizowane, oraz usunięcie tracków, które przekroczyły max_age.

#### 2.4. Obliczanie IoU
Metoda **calculate_iou** oblicza współczynnik IoU pomiędzy dwoma prostokątami ograniczającymi, który jest używany do określania podobieństwa między śledzonymi obiektami a nowymi detekcjami.

#### 2.5. Wizualizacja tracków
Funkcja **visualize_tracks** umożliwia wizualizację detekcji i tracków poprzez rysowanie prostokątów na klatkach.

#### 2.6. Wykonanie śledzenia pieszych
Funkcja **track_pedestrians** zarządza procesem śledzenia we wszystkich klatkach w zestawie danych.
- Kroki:
  - Wczytanie prostokątów ograniczających z zestawu danych.
  - Aktualizacja tracków za pomocą nowych detekcji dla każdej klatki.
  - Zapis wyników procesu śledzenia.
  - Obliczanie i wyświetlanie całkowitego oraz średniego czasu przetwarzania.

#### 2.7. Ocena dokładności
Dokładność trackera jest oceniana za pomocą dwóch metryk:
- Porównanie liczby osób występujących na danej klatce obrazu do obliczonej przez program liczby osób.
- Porównanie posortowanych wyników tracków z posortowanymi wektorami ground truth dla każdej klatki.

### 3. Użycie
Aby uruchomić algorytm śledzenia, należy wykonać poniższe polecenie:
```python main.py <path_to_dataset>```

Zestaw danych powinien zawierać katalog **frames** z plikami obrazów oraz plik **bboxes.txt** z informacjami o prostokątach ograniczających.

### 4. Metryki wydajności
Wydajność trackera jest mierzona poprzez:
- Całkowity czas wykonania.
- Średni czas przetwarzania na klatkę.
- Dokładność na podstawie prostokątów ograniczających (**bboxes.txt**).
- Dokładność na podstawie wektorów ground truth (**bboxes_gt_vector_filtered.txt**).

Poniżej uzyskane wartości dla tego projektu.

![Metryki](https://github.com/div-57/SIwR-lab-project/blob/main/img/metryki_wydajnosci.png)

### 5. Graf modelu

![Graf](https://github.com/div-57/SIwR-lab-project/blob/main/img/graf.png)


*Autorstwo: Waszkowiak Michał, Woźniak Dawid*
