from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableView, QVBoxLayout, QWidget,
                             QPushButton, QFileDialog, QHeaderView, QHBoxLayout, QLabel,
                             QLineEdit, QComboBox, QGroupBox, QFormLayout, QTextEdit,
                             QMessageBox, QProgressBar, QMenu, QShortcut)
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QKeySequence, QFont, QClipboard
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QMimeData, QItemSelection, QItemSelectionRange
import pandas as pd
import sys
import codecs
from datetime import datetime
import os
import numpy as np
import csv

# Константы для приложения
MAX_DISPLAY_ROWS = 100000
MAX_DISPLAY_COLUMNS = 20
DISPLAY_LIMIT_LARGE_FILES = 1000000


class CSVProcessingThread(QThread):
    """Поток для обработки CSV файла с поддержкой кириллицы"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            # Определение кодировки с поддержкой кириллицы
            self.progress.emit(10)
            encoding = self.detect_encoding_with_cyrillic(self.file_path)

            self.progress.emit(30)
            # Определение разделителя
            delimiter = self.detect_delimiter(self.file_path, encoding)

            self.progress.emit(50)
            # Загрузка данных с правильной кодировкой
            df = self.load_csv_with_fallback(self.file_path, encoding, delimiter)

            self.progress.emit(80)
            # Очистка данных с сохранением кириллицы
            df = self.clean_data_with_cyrillic(df)

            self.progress.emit(100)
            self.finished.emit(df)

        except Exception as e:
            self.error.emit(str(e))

    def detect_encoding_with_cyrillic(self, file_path):
        """Определение кодировки с поддержкой кириллицы"""
        cyrillic_encodings = ['utf-8-sig', 'windows-1251', 'cp1251', 'koi8-r', 'iso-8859-5', 'utf-8']

        for encoding in cyrillic_encodings:
            try:
                # Пробуем прочитать первые несколько строк
                with codecs.open(file_path, 'r', encoding=encoding, errors='strict') as f:
                    lines = [f.readline() for _ in range(5)]

                # Проверяем, что файл содержит русские буквы
                russian_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
                content = ''.join(lines)
                if any(char in content for char in russian_chars):
                    return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception:
                continue

        return 'utf-8'

    def detect_delimiter(self, file_path, encoding):
        """Определение разделителя"""
        delimiters = [';', ',', '\t', '|']
        try:
            with codecs.open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                sample_lines = [f.readline() for _ in range(10)]

            delimiter_counts = {}
            for delim in delimiters:
                count = sum(line.count(delim) for line in sample_lines)
                delimiter_counts[delim] = count

            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)

            # Если разделитель не найден, пробуем определить по расширению
            if delimiter_counts[best_delimiter] == 0:
                if file_path.endswith('.tsv'):
                    return '\t'
                elif file_path.endswith('.csv'):
                    # Для CSV чаще используется запятая или точка с запятой
                    # Проверяем, есть ли запятые в данных
                    comma_count = sum(line.count(',') for line in sample_lines)
                    semicolon_count = sum(line.count(';') for line in sample_lines)
                    return ',' if comma_count > semicolon_count else ';'
                else:
                    return ';'
            return best_delimiter
        except:
            return ';'

    def load_csv_with_fallback(self, file_path, encoding, delimiter):
        """Загрузка CSV с механизмом fallback для кодировок"""
        fallback_encodings = ['utf-8-sig', 'windows-1251', 'cp1251', 'utf-8', 'latin1']

        for enc in [encoding] + fallback_encodings:
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=enc,
                    delimiter=delimiter,
                    on_bad_lines='skip',
                    low_memory=False,
                    dtype=str,
                    skipinitialspace=True
                )

                # Проверяем, что загрузились данные и есть русские символы
                if not df.empty and len(df.columns) > 0:
                    # Проверяем несколько первых строк на наличие русских символов
                    sample_text = ''
                    for col in df.columns[:3]:
                        if col and isinstance(col, str):
                            sample_text += col

                    # Если есть русские символы или данные загружены - возвращаем
                    if any(c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
                           for c in sample_text):
                        return df
                    elif not df.empty:
                        return df
            except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError) as e:
                continue
            except Exception as e:
                continue

        # Если все попытки не удались, пробуем с engine='python'
        try:
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                delimiter=delimiter,
                on_bad_lines='skip',
                low_memory=False,
                dtype=str,
                skipinitialspace=True,
                engine='python'
            )
            return df
        except:
            raise Exception("Не удалось загрузить файл. Проверьте кодировку и разделитель.")

    def clean_data_with_cyrillic(self, df):
        """Очистка данных с сохранением кириллицы"""
        try:
            # Очищаем заголовки столбцов
            original_columns = df.columns.tolist()
            cleaned_columns = []

            for i, col in enumerate(original_columns):
                if isinstance(col, str):
                    # Убираем лишние пробелы, но сохраняем кириллицу
                    cleaned = col.strip()
                    # Удаляем непечатаемые символы, но оставляем русские буквы
                    cleaned = ''.join(char for char in cleaned if
                                      char.isprintable() or char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ')
                    cleaned_columns.append(cleaned)
                else:
                    # Если заголовок не строка, преобразуем
                    cleaned_columns.append(f"Column_{i}")

            df.columns = cleaned_columns

            # Для каждого столбца сохраняем данные как строки
            for col in df.columns:
                # Преобразуем в строку, сохраняя кириллицу
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else "")

            # Замена NaN на пустые строки
            df = df.fillna('')

            return df

        except Exception as e:
            print(f"Ошибка при очистке данных: {e}")
            return df


class CSVViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_df = None
        self.current_file_path = None
        self.processing_thread = None
        self.initUI()
        self.setup_shortcuts()
        self.setup_context_menu()

    def initUI(self):
        self.setWindowTitle("CSV Converter")
        self.setGeometry(100, 100, 1400, 700)

        # Создаем центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Основной горизонтальный layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # Левая панель - таблица
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)

        self.table_view = QTableView()
        self.model = QStandardItemModel()
        self.table_view.setModel(self.model)

        # Упрощенные настройки таблицы
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        self.table_view.setStyleSheet("""
            QTableView {
                background-color: #f5f5f5;
                font-size: 10px;
            }
            QTableView::item {
                padding: 2px;
            }
            QTableView::item:selected {
                background-color: #0078d7;
                color: white;
            }
        """)

        # Включаем контекстное меню
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.customContextMenuRequested.connect(self.show_context_menu)

        left_layout.addWidget(self.table_view)
        main_layout.addWidget(left_widget, stretch=3)

        # Правая панель - элементы управления
        right_widget = QWidget()
        right_widget.setFixedWidth(350)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setSpacing(10)

        # Группа настроек
        connection_group = QGroupBox("Настройки")
        connection_layout = QFormLayout(connection_group)

        self.owner_input = QLineEdit()
        connection_layout.addRow("Владелец:", self.owner_input)

        self.port_input = QLineEdit()
        connection_layout.addRow("Порт:", self.port_input)

        self.utc_combo = QComboBox()
        self.populate_timezones()
        connection_layout.addRow("Часовой пояс:", self.utc_combo)

        self.object_input = QLineEdit()
        connection_layout.addRow("Объект:", self.object_input)

        # Информация
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)

        self.row_count_label = QLabel("Файл не загружен")
        self.row_count_label.setStyleSheet("color: #666;")
        info_layout.addWidget(self.row_count_label)

        self.file_info_label = QLabel("")
        self.file_info_label.setStyleSheet("color: #666; font-size: 9px;")
        info_layout.addWidget(self.file_info_label)

        connection_layout.addRow("Информация:", info_widget)

        right_layout.addWidget(connection_group)

        # Кнопки
        buttons_widget = QWidget()
        buttons_layout = QVBoxLayout(buttons_widget)
        buttons_layout.setSpacing(5)

        self.load_btn = QPushButton("Загрузить CSV")
        self.load_btn.clicked.connect(self.load_csv)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        buttons_layout.addWidget(self.load_btn)

        self.generate_btn = QPushButton("Сгенерировать Template")
        self.generate_btn.clicked.connect(self.generate_template)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #107c10;
                color: white;
                padding: 8px;
                border: none;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0e6b0e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        buttons_layout.addWidget(self.generate_btn)

        right_layout.addWidget(buttons_widget)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)

        # Журнал сообщений
        message_group = QGroupBox("Журнал")
        message_layout = QVBoxLayout(message_group)

        self.message_display = QTextEdit()
        self.message_display.setMaximumHeight(150)
        self.message_display.setReadOnly(True)
        message_layout.addWidget(self.message_display)

        right_layout.addWidget(message_group)
        right_layout.addStretch()

        main_layout.addWidget(right_widget)

    def setup_shortcuts(self):
        """Настройка горячих клавиш"""
        # Загрузка файла
        QShortcut(QKeySequence("Ctrl+O"), self, self.load_csv)

        # Генерация template
        QShortcut(QKeySequence("Ctrl+G"), self, self.generate_template)

        # Копирование
        QShortcut(QKeySequence("Ctrl+C"), self, self.copy_selected_cells)

        # Вставка
        QShortcut(QKeySequence("Ctrl+V"), self, self.paste_to_cells)

        # Выделить все
        QShortcut(QKeySequence("Ctrl+A"), self, self.select_all_cells)

        # Удаление выделенных ячеек
        QShortcut(QKeySequence("Delete"), self, self.clear_selected_cells)

        # Расширение выделения вниз
        QShortcut(QKeySequence("Ctrl+Shift+Down"), self, self.extend_selection_down)

        # Расширение выделения вверх
        QShortcut(QKeySequence("Ctrl+Shift+Up"), self, self.extend_selection_up)

        # Расширение выделения вправо
        QShortcut(QKeySequence("Ctrl+Shift+Right"), self, self.extend_selection_right)

        # Расширение выделения влево
        QShortcut(QKeySequence("Ctrl+Shift+Left"), self, self.extend_selection_left)

    def setup_context_menu(self):
        """Настройка контекстного меню для таблицы"""
        self.context_menu = QMenu(self.table_view)

        copy_action = self.context_menu.addAction("Копировать (Ctrl+C)")
        copy_action.triggered.connect(self.copy_selected_cells)

        paste_action = self.context_menu.addAction("Вставить (Ctrl+V)")
        paste_action.triggered.connect(self.paste_to_cells)

        self.context_menu.addSeparator()

        select_all_action = self.context_menu.addAction("Выделить все (Ctrl+A)")
        select_all_action.triggered.connect(self.select_all_cells)

        clear_action = self.context_menu.addAction("Очистить (Delete)")
        clear_action.triggered.connect(self.clear_selected_cells)

        self.context_menu.addSeparator()

        # Добавляем пункты для расширения выделения
        extend_down_action = self.context_menu.addAction("Расширить выделение вниз (Ctrl+Shift+↓)")
        extend_down_action.triggered.connect(self.extend_selection_down)

        extend_up_action = self.context_menu.addAction("Расширить выделение вверх (Ctrl+Shift+↑)")
        extend_up_action.triggered.connect(self.extend_selection_up)

        extend_right_action = self.context_menu.addAction("Расширить выделение вправо (Ctrl+Shift+→)")
        extend_right_action.triggered.connect(self.extend_selection_right)

        extend_left_action = self.context_menu.addAction("Расширить выделение влево (Ctrl+Shift+←)")
        extend_left_action.triggered.connect(self.extend_selection_left)

        self.context_menu.addSeparator()

        fill_down_action = self.context_menu.addAction("Заполнить вниз")
        fill_down_action.triggered.connect(self.fill_down)

        fill_right_action = self.context_menu.addAction("Заполнить вправо")
        fill_right_action.triggered.connect(self.fill_right)

    def show_context_menu(self, position):
        """Показать контекстное меню"""
        self.context_menu.exec_(self.table_view.mapToGlobal(position))

    def populate_timezones(self):
        """Заполнение списка часовых поясов"""
        timezones = [
            "UTC-12", "UTC-11", "UTC-10", "UTC-9", "UTC-8", "UTC-7", "UTC-6", "UTC-5",
            "UTC-4", "UTC-3", "UTC-2", "UTC-1", "UTC±0", "UTC+1", "UTC+2", "UTC+3",
            "UTC+4", "UTC+5", "UTC+6", "UTC+7", "UTC+8", "UTC+9", "UTC+10", "UTC+11", "UTC+12"
        ]
        self.utc_combo.addItems(timezones)
        self.utc_combo.setCurrentText("UTC+3")  # По умолчанию Москва

    def add_message(self, message, message_type="info"):
        """Добавить сообщение в область сообщений"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if message_type == "success":
            color = "green"
            prefix = "✓"
        elif message_type == "error":
            color = "red"
            prefix = "✗"
        else:
            color = "blue"
            prefix = "ℹ"

        formatted_message = f'<span style="color: gray;">[{timestamp}]</span> ' \
                            f'<span style="color: {color}; font-weight: bold;">{prefix}</span> {message}<br>'

        current_text = self.message_display.toHtml()
        self.message_display.setHtml(formatted_message + current_text)

    # ========== Методы для работы с копипастом ==========

    def copy_selected_cells(self):
        """Копирование выделенных ячеек в буфер обмена"""
        selection = self.table_view.selectionModel()
        if not selection.hasSelection():
            self.add_message("Нет выделенных значений для копирования", "warning")
            return

        selected_indexes = selection.selectedIndexes()
        if not selected_indexes:
            return

        # Определяем границы выделения
        rows = sorted(set(index.row() for index in selected_indexes))
        cols = sorted(set(index.column() for index in selected_indexes))

        min_row = min(rows)
        max_row = max(rows)
        min_col = min(cols)
        max_col = max(cols)

        # Создаем данные для копирования
        copied_data = []
        for row in range(min_row, max_row + 1):
            row_data = []
            for col in range(min_col, max_col + 1):
                index = self.model.index(row, col)
                item = self.model.itemFromIndex(index)
                if item:
                    row_data.append(item.text())
                else:
                    row_data.append("")
            copied_data.append(row_data)

        # Формируем текстовое представление (табуляция между столбцами, новая строка между строками)
        text_data = "\n".join("\t".join(row) for row in copied_data)

        # Копируем в буфер обмена
        clipboard = QApplication.clipboard()
        clipboard.setText(text_data)

        # Также сохраняем данные в формате CSV для лучшей совместимости
        csv_data = "\n".join(",".join(f'"{cell}"' for cell in row) for row in copied_data)

        mime_data = QMimeData()
        mime_data.setText(text_data)
        mime_data.setData("text/csv", csv_data.encode('utf-8'))
        clipboard.setMimeData(mime_data)

        self.add_message(f"Скопировано {len(selected_indexes)} значений", "success")

    def paste_to_cells(self):
        """Вставка данных из буфера обмена в таблицу"""
        if self.current_df is None:
            self.add_message("Нет загруженных данных для вставки", "warning")
            return

        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()

        if not mime_data.hasText():
            self.add_message("Буфер обмена пуст или не содержит текста", "warning")
            return

        text_data = clipboard.text()
        if not text_data.strip():
            self.add_message("Буфер обмена содержит пустой текст", "warning")
            return

        try:
            # Определяем текущую выделенную ячейку
            current_index = self.table_view.currentIndex()
            if not current_index.isValid():
                current_index = self.model.index(0, 0)

            start_row = current_index.row()
            start_col = current_index.column()

            # Парсим данные из буфера обмена
            pasted_rows = text_data.strip().split('\n')
            pasted_data = []

            for row in pasted_rows:
                # Пробуем разные разделители
                if '\t' in row:
                    cells = row.split('\t')
                elif ';' in row:
                    cells = row.split(';')
                else:
                    cells = row.split(',')

                # Убираем кавычки если они есть
                cells = [cell.strip('"\' ') for cell in cells]
                pasted_data.append(cells)

            # Определяем размер вставляемых данных
            rows_to_paste = len(pasted_data)
            cols_to_paste = max(len(row) for row in pasted_data) if pasted_data else 0

            # Проверяем границы таблицы
            max_row = self.model.rowCount()
            max_col = self.model.columnCount()

            if start_row + rows_to_paste > max_row or start_col + cols_to_paste > max_col:
                reply = QMessageBox.question(
                    self,
                    "Расширение таблицы",
                    "Вставка данных выходит за границы таблицы. Расширить таблицу?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    # Добавляем строки если нужно
                    if start_row + rows_to_paste > max_row:
                        rows_to_add = start_row + rows_to_paste - max_row
                        self.model.insertRows(max_row, rows_to_add)

                    # Добавляем столбцы если нужно
                    if start_col + cols_to_paste > max_col:
                        cols_to_add = start_col + cols_to_paste - max_col
                        self.model.insertColumns(max_col, cols_to_add)
                else:
                    # Обрезаем данные если выходим за границы
                    rows_to_paste = min(rows_to_paste, max_row - start_row)
                    cols_to_paste = min(cols_to_paste, max_col - start_col)

            # Вставляем данные
            cells_pasted = 0
            for i in range(rows_to_paste):
                for j in range(min(cols_to_paste, len(pasted_data[i]))):
                    row_idx = start_row + i
                    col_idx = start_col + j

                    if row_idx < self.model.rowCount() and col_idx < self.model.columnCount():
                        cell_value = pasted_data[i][j]
                        item = self.model.item(row_idx, col_idx)

                        if item:
                            item.setText(cell_value)
                        else:
                            item = QStandardItem(cell_value)
                            self.model.setItem(row_idx, col_idx, item)

                        cells_pasted += 1

            # Обновляем DataFrame
            self.update_dataframe_from_model()

            self.add_message(f"Вставлено {cells_pasted} значений", "success")

        except Exception as e:
            self.add_message(f"Ошибка при вставке данных: {str(e)}", "error")
            QMessageBox.warning(self, "Ошибка вставки", f"Не удалось вставить данные:\n\n{str(e)}")

    def select_all_cells(self):
        """Выделить все ячейки в таблице"""
        self.table_view.selectAll()
        row_count = self.model.rowCount()
        col_count = self.model.columnCount()
        self.add_message(f"Выделено всех значений: {row_count * col_count}", "info")

    def clear_selected_cells(self):
        """Очистить содержимое выделенных ячеек"""
        selection = self.table_view.selectionModel()
        if not selection.hasSelection():
            self.add_message("Нет выделенных значений для очистки", "warning")
            return

        selected_indexes = selection.selectedIndexes()
        if not selected_indexes:
            return

        reply = QMessageBox.question(
            self,
            "Очистка ячеек",
            f"Очистить {len(selected_indexes)} выделенных ячеек?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            for index in selected_indexes:
                item = self.model.itemFromIndex(index)
                if item:
                    item.setText("")

            # Обновляем DataFrame
            self.update_dataframe_from_model()

            self.add_message(f"Очищено {len(selected_indexes)} значений", "success")

    def extend_selection_down(self):
        """Расширить выделение вниз"""
        try:
            selection_model = self.table_view.selectionModel()
            if not selection_model.hasSelection():
                # Если нет выделения, начинаем с текущей ячейки
                current_index = self.table_view.currentIndex()
                if not current_index.isValid():
                    # Если нет текущей ячейки, берем первую
                    if self.model.rowCount() > 0 and self.model.columnCount() > 0:
                        current_index = self.model.index(0, 0)
                    else:
                        return

                # Выделяем одну ячейку
                selection = QItemSelection()
                selection.select(current_index, current_index)
                selection_model.select(selection, selection_model.ClearAndSelect)

            # Получаем текущее выделение
            selected_indexes = selection_model.selectedIndexes()
            if not selected_indexes:
                return

            # Определяем границы выделения
            rows = sorted(set(index.row() for index in selected_indexes))
            cols = sorted(set(index.column() for index in selected_indexes))

            min_row = min(rows)
            max_row = max(rows)
            min_col = min(cols)
            max_col = max(cols)

            # Проверяем, можно ли расширить вниз
            last_row = self.model.rowCount() - 1
            if max_row < last_row:
                # Создаем новое выделение
                new_selection = QItemSelection()

                # Создаем диапазон выделения
                top_left = self.model.index(min_row, min_col)
                bottom_right = self.model.index(last_row, max_col)
                new_selection.select(top_left, bottom_right)

                # Устанавливаем новое выделение
                selection_model.select(new_selection, selection_model.ClearAndSelect)

                # Получаем количество выделенных ячеек
                rows_selected = last_row - min_row + 1
                cols_selected = max_col - min_col + 1
                total_selected = rows_selected * cols_selected

                self.add_message(f"Выделено {total_selected} значений", "info")
            else:
                self.add_message("Невозможно расширить выделение вниз - достигнута последняя строка", "warning")

        except Exception as e:
            self.add_message(f"Ошибка при расширении выделения вниз: {str(e)}", "error")

    def extend_selection_up(self):
        """Расширить выделение вверх"""
        try:
            selection_model = self.table_view.selectionModel()
            if not selection_model.hasSelection():
                # Если нет выделения, начинаем с текущей ячейки
                current_index = self.table_view.currentIndex()
                if not current_index.isValid():
                    # Если нет текущей ячейки, берем первую
                    if self.model.rowCount() > 0 and self.model.columnCount() > 0:
                        current_index = self.model.index(0, 0)
                    else:
                        return

                # Выделяем одну ячейку
                selection = QItemSelection()
                selection.select(current_index, current_index)
                selection_model.select(selection, selection_model.ClearAndSelect)

            # Получаем текущее выделение
            selected_indexes = selection_model.selectedIndexes()
            if not selected_indexes:
                return

            # Определяем границы выделения
            rows = sorted(set(index.row() for index in selected_indexes))
            cols = sorted(set(index.column() for index in selected_indexes))

            min_row = min(rows)
            max_row = max(rows)
            min_col = min(cols)
            max_col = max(cols)

            # Проверяем, можно ли расширить вверх
            if min_row > 0:
                # Создаем новое выделение
                new_selection = QItemSelection()

                # Создаем диапазон выделения
                top_left = self.model.index(0, min_col)
                bottom_right = self.model.index(max_row, max_col)
                new_selection.select(top_left, bottom_right)

                # Устанавливаем новое выделение
                selection_model.select(new_selection, selection_model.ClearAndSelect)

                # Получаем количество выделенных ячеек
                rows_selected = max_row - 0 + 1  # от 0 до max_row
                cols_selected = max_col - min_col + 1
                total_selected = rows_selected * cols_selected

                self.add_message(f"Выделено {total_selected} значений", "info")
            else:
                self.add_message("Невозможно выделить содержимое - достигнута первая строка", "warning")

        except Exception as e:
            self.add_message(f"Ошибка при выделении вверх: {str(e)}", "error")

    def extend_selection_right(self):
        """Расширить выделение вправо"""
        try:
            selection_model = self.table_view.selectionModel()
            if not selection_model.hasSelection():
                # Если нет выделения, начинаем с текущей ячейки
                current_index = self.table_view.currentIndex()
                if not current_index.isValid():
                    # Если нет текущей ячейки, берем первую
                    if self.model.rowCount() > 0 and self.model.columnCount() > 0:
                        current_index = self.model.index(0, 0)
                    else:
                        return

                # Выделяем одну ячейку
                selection = QItemSelection()
                selection.select(current_index, current_index)
                selection_model.select(selection, selection_model.ClearAndSelect)

            # Получаем текущее выделение
            selected_indexes = selection_model.selectedIndexes()
            if not selected_indexes:
                return

            # Определяем границы выделения
            rows = sorted(set(index.row() for index in selected_indexes))
            cols = sorted(set(index.column() for index in selected_indexes))

            min_row = min(rows)
            max_row = max(rows)
            min_col = min(cols)
            max_col = max(cols)

            # Проверяем, можно ли расширить вправо
            last_col = self.model.columnCount() - 1
            if max_col < last_col:
                # Создаем новое выделение
                new_selection = QItemSelection()

                # Создаем диапазон выделения
                top_left = self.model.index(min_row, min_col)
                bottom_right = self.model.index(max_row, last_col)
                new_selection.select(top_left, bottom_right)

                # Устанавливаем новое выделение
                selection_model.select(new_selection, selection_model.ClearAndSelect)

                # Получаем количество выделенных ячеек
                rows_selected = max_row - min_row + 1
                cols_selected = last_col - min_col + 1
                total_selected = rows_selected * cols_selected

                self.add_message(f"Выделено {total_selected} значений", "info")
            else:
                self.add_message("Невозможно расширить выделение вправо - достигнут последний столбец", "warning")

        except Exception as e:
            self.add_message(f"Ошибка при расширении выделения вправо: {str(e)}", "error")

    def extend_selection_left(self):
        """Расширить выделение влево"""
        try:
            selection_model = self.table_view.selectionModel()
            if not selection_model.hasSelection():
                # Если нет выделения, начинаем с текущей ячейки
                current_index = self.table_view.currentIndex()
                if not current_index.isValid():
                    # Если нет текущей ячейки, берем первую
                    if self.model.rowCount() > 0 and self.model.columnCount() > 0:
                        current_index = self.model.index(0, 0)
                    else:
                        return

                # Выделяем одну ячейку
                selection = QItemSelection()
                selection.select(current_index, current_index)
                selection_model.select(selection, selection_model.ClearAndSelect)

            # Получаем текущее выделение
            selected_indexes = selection_model.selectedIndexes()
            if not selected_indexes:
                return

            # Определяем границы выделения
            rows = sorted(set(index.row() for index in selected_indexes))
            cols = sorted(set(index.column() for index in selected_indexes))

            min_row = min(rows)
            max_row = max(rows)
            min_col = min(cols)
            max_col = max(cols)

            # Проверяем, можно ли расширить влево
            if min_col > 0:
                # Создаем новое выделение
                new_selection = QItemSelection()

                # Создаем диапазон выделения
                top_left = self.model.index(min_row, 0)
                bottom_right = self.model.index(max_row, max_col)
                new_selection.select(top_left, bottom_right)

                # Устанавливаем новое выделение
                selection_model.select(new_selection, selection_model.ClearAndSelect)

                rows_selected = max_row - min_row + 1
                cols_selected = max_col - 0 + 1  # от 0 до max_col
                total_selected = rows_selected * cols_selected

                self.add_message(f"Выделено {total_selected} значений", "info")
            else:
                self.add_message("Невозможно расширить выделение влево - достигнут первый столбец", "warning")

        except Exception as e:
            self.add_message(f"Ошибка при расширении выделения влево: {str(e)}", "error")

    def fill_down(self):
        """Заполнить ячейки вниз значением из первой выделенной строки"""
        selection = self.table_view.selectionModel()
        if not selection.hasSelection():
            self.add_message("Нет выделенных ячеек для заполнения", "warning")
            return

        selected_indexes = selection.selectedIndexes()
        if not selected_indexes:
            return

        # Группируем по столбцам
        columns = {}
        for index in selected_indexes:
            col = index.column()
            if col not in columns:
                columns[col] = []
            columns[col].append(index.row())

        # Для каждого столбца заполняем вниз
        cells_filled = 0
        for col, rows in columns.items():
            if not rows:
                continue

            sorted_rows = sorted(rows)
            top_row = sorted_rows[0]

            # Получаем значение из верхней ячейки
            top_index = self.model.index(top_row, col)
            top_item = self.model.itemFromIndex(top_index)
            if not top_item:
                continue

            fill_value = top_item.text()

            # Заполняем все ячейки ниже
            for row in sorted_rows[1:]:
                item = self.model.item(row, col)
                if item:
                    item.setText(fill_value)
                else:
                    item = QStandardItem(fill_value)
                    self.model.setItem(row, col, item)
                cells_filled += 1

        if cells_filled > 0:
            self.update_dataframe_from_model()
            self.add_message(f"Заполнено вниз {cells_filled} значений", "success")

    def fill_right(self):
        """Заполнить ячейки вправо значением из первого выделенного столбца"""
        selection = self.table_view.selectionModel()
        if not selection.hasSelection():
            self.add_message("Нет выделенных значений для заполнения", "warning")
            return

        selected_indexes = selection.selectedIndexes()
        if not selected_indexes:
            return

        # Группируем по строкам
        rows = {}
        for index in selected_indexes:
            row = index.row()
            if row not in rows:
                rows[row] = []
            rows[row].append(index.column())

        # Для каждой строки заполняем вправо
        cells_filled = 0
        for row, cols in rows.items():
            if not cols:
                continue

            sorted_cols = sorted(cols)
            left_col = sorted_cols[0]

            # Получаем значение из левой ячейки
            left_index = self.model.index(row, left_col)
            left_item = self.model.itemFromIndex(left_index)
            if not left_item:
                continue

            fill_value = left_item.text()

            # Заполняем все ячейки справа
            for col in sorted_cols[1:]:
                item = self.model.item(row, col)
                if item:
                    item.setText(fill_value)
                else:
                    item = QStandardItem(fill_value)
                    self.model.setItem(row, col, item)
                cells_filled += 1

        if cells_filled > 0:
            self.update_dataframe_from_model()
            self.add_message(f"Заполнено вправо {cells_filled} значений", "success")

    def update_dataframe_from_model(self):
        """Обновление DataFrame на основе изменений в модели"""
        if self.current_df is None:
            return

        try:
            # Получаем данные из модели
            rows = self.model.rowCount()
            cols = self.model.columnCount()

            # Создаем новый DataFrame с актуальными данными
            new_data = []
            for row in range(rows):
                row_data = []
                for col in range(cols):
                    item = self.model.item(row, col)
                    if item:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                new_data.append(row_data)

            # Обновляем current_df если размеры совпадают
            if rows == len(self.current_df) and cols == len(self.current_df.columns):
                for i in range(rows):
                    for j in range(cols):
                        self.current_df.iat[i, j] = new_data[i][j]

            self.add_message("Данные обновлены", "info")

        except Exception as e:
            self.add_message(f"Ошибка обновления DataFrame: {str(e)}", "error")

    # ========== Остальные методы класса ==========

    def load_csv(self):
        """Загрузка CSV файла"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите CSV файл",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        self.current_file_path = file_path
        filename = os.path.basename(file_path)

        # Показываем прогресс-бар и отключаем кнопки
        self.progress_bar.setVisible(True)
        self.load_btn.setEnabled(False)
        self.generate_btn.setEnabled(False)
        self.add_message(f"Загрузка файла: {filename}", "info")

        # Останавливаем предыдущий поток, если он есть
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()

        # Запускаем обработку в отдельном потоке
        self.processing_thread = CSVProcessingThread(file_path)
        self.processing_thread.progress.connect(self.progress_bar.setValue)
        self.processing_thread.finished.connect(self.on_csv_loaded)
        self.processing_thread.error.connect(self.on_csv_error)
        self.processing_thread.start()

    def on_csv_loaded(self, df):
        """Обработка успешной загрузки CSV"""
        try:
            # Ограничиваем количество отображаемых строк для больших файлов
            if len(df) > DISPLAY_LIMIT_LARGE_FILES:
                reply = QMessageBox.information(
                    self, "Большой файл",
                    f"Файл содержит {len(df):,} строк. "
                    f"Показать только первые {DISPLAY_LIMIT_LARGE_FILES} строк для производительности?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    display_df = df.head(DISPLAY_LIMIT_LARGE_FILES)
                    self.add_message(f"Показано {DISPLAY_LIMIT_LARGE_FILES} из {len(df)} значений", "info")
                else:
                    display_df = df
            else:
                display_df = df

            self.current_df = df
            self.display_data(display_df)

            # Автозаполнение
            self.auto_fill_fields(df)

            # Обновление информации
            row_count = len(df)
            self.row_count_label.setText(f"Строк: {row_count:,}")
            self.row_count_label.setStyleSheet("color: green; font-weight: bold;")

            filename = os.path.basename(self.current_file_path)
            self.file_info_label.setText(f"{filename}\n({(os.path.getsize(self.current_file_path) / 1024):.0f} КБ)")

            # Активация кнопки
            self.generate_btn.setEnabled(True)

            # Скрываем прогресс-бар
            self.progress_bar.setVisible(False)
            self.load_btn.setEnabled(True)

            self.add_message(f"Загружено {row_count} устройств", "success")

            # Проверка обязательных столбцов
            self.check_required_columns(df)

        except Exception as e:
            self.on_csv_error(f"Ошибка при обработке данных: {str(e)}")

    def on_csv_error(self, error_message):
        """Обработка ошибки"""
        self.progress_bar.setVisible(False)
        self.load_btn.setEnabled(True)

        self.add_message(f"Ошибка: {error_message}", "error")
        QMessageBox.critical(self, "Ошибка загрузки",
                             f"Не удалось загрузить файл:\n\n{error_message}")

    def check_required_columns(self, df):
        """Проверка обязательных столбцов"""
        required = ['DevEUI', 'Модель ПУ', 'Серийный номер']
        missing = [col for col in required if col not in df.columns]

        if missing:
            self.add_message(f"Отсутствуют столбцы: {', '.join(missing)}", "warning")

    def auto_fill_fields(self, df):
        """Автозаполнение полей"""
        try:
            # Владелец
            if 'Владелец' in df.columns:
                owners = df['Владелец'].dropna().unique()
                if len(owners) > 0:
                    self.owner_input.setText(str(owners[0]))

            # Объект
            if 'Объект' in df.columns:
                objects = df['Объект'].dropna().unique()
                if len(objects) > 0:
                    self.object_input.setText(str(objects[0]))

        except Exception as e:
            self.add_message(f"Ошибка автозаполнения: {e}", "warning")

    def display_data(self, df):
        """Безопасное отображение данных"""
        try:
            self.model.clear()

            # Ограничиваем количество столбцов для отображения
            if len(df.columns) > MAX_DISPLAY_COLUMNS:
                df_display = df.iloc[:, :MAX_DISPLAY_COLUMNS]
                self.add_message(f"Показано {MAX_DISPLAY_COLUMNS} из {len(df.columns)} столбцов", "info")
            else:
                df_display = df

            # Устанавливаем заголовки
            self.model.setHorizontalHeaderLabels(df_display.columns.tolist())

            # Заполняем данными (ограничиваем количество строк для отображения)
            max_display_rows = min(MAX_DISPLAY_ROWS, len(df_display))
            for row_idx in range(max_display_rows):
                row_items = []
                for col_idx, value in enumerate(df_display.iloc[row_idx]):
                    # Сохраняем точное строковое представление
                    display_value = str(value) if pd.notna(value) else ""
                    item = QStandardItem(display_value)
                    row_items.append(item)
                self.model.appendRow(row_items)

            # Настраиваем ширину столбцов
            header = self.table_view.horizontalHeader()
            for col in range(df_display.shape[1]):
                header.setSectionResizeMode(col, QHeaderView.ResizeToContents)

            self.table_view.setWindowTitle(f"Загружено: {len(df)} строк")

        except Exception as e:
            self.add_message(f"Ошибка отображения данных: {e}", "error")

    def generate_template(self):
        """Генерация template файла"""
        if self.current_df is None:
            self.add_message("Нет данных для генерации", "error")
            return

        try:
            # Получаем настройки
            owner = self.owner_input.text().strip() or "Не задано"
            port = self.port_input.text().strip()
            utc = self.convert_utc_format(self.utc_combo.currentText())
            obj = self.object_input.text().strip()

            # Создаем template
            template_data = []

            for idx, row in self.current_df.iterrows():
                try:
                    # Получаем значения с сохранением формата
                    deveui = str(row.get('DevEUI', '')).strip()
                    if not deveui:
                        continue

                    model_name = str(row.get('Модель ПУ', '')).strip()
                    firmware = str(row.get('Прошивка', '')).strip()
                    model = self.convert_model_name(model_name, firmware)

                    # Серийный номер с сохранением ведущих нулей
                    serial = str(row.get('Серийный номер', '')).strip()

                    template_row = {
                        'DevEui': deveui,
                        'Model': model,
                        'SerialNumber': serial,
                        'SetupDate': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'UTC': utc,
                        'Network': 'sophrosyne',
                        'Owner': owner,
                        'User': '',
                        'Application': 'develop',
                        'Place': obj,
                        'Port': port,
                        'PhysicalAddress': '',
                        'LLS': '',
                        'HLS': ''
                    }
                    template_data.append(template_row)

                except Exception as e:
                    self.add_message(f"Ошибка в строке {idx + 1}: {e}", "warning")

            if not template_data:
                self.add_message("Нет данных для генерации", "error")
                return

            # Создаем DataFrame
            template_df = pd.DataFrame(template_data)

            # Сохраняем
            default_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            default_name = f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Сохранить Template",
                os.path.join(default_dir, default_name),
                "CSV Files (*.csv);;All Files (*)"
            )

            if file_path:
                # Сохраняем как текст для сохранения ведущих нулей
                template_df.to_csv(file_path, index=False, encoding='utf-8-sig', sep=',')

                success_msg = f"Создано {len(template_data)} записей в файл: {os.path.basename(file_path)}"
                self.add_message(success_msg, "success")
                QMessageBox.information(self, "Успех", success_msg)

        except Exception as e:
            self.add_message(f"Ошибка генерации: {e}", "error")
            QMessageBox.critical(self, "Ошибка", f"Ошибка генерации template: Возможно файл открыт в другой программе!\n\n{str(e)}")

    def convert_utc_format(self, utc_str):
        """Конвертация UTC"""
        try:
            if utc_str.startswith('UTC+'):
                hours = utc_str[4:]
                return f"+{int(hours):02d}:00"
            elif utc_str.startswith('UTC-'):
                hours = utc_str[4:]
                return f"-{int(hours):02d}:00"
            elif utc_str == 'UTC±0':
                return "+00:00"
            else:
                return "+03:00"
        except:
            return "+03:00"

    def convert_model_name(self, model_name, firmware=None):
        """Конвертирует название модели в формат template с учетом прошивки"""
        if not isinstance(model_name, str):
            model_name = str(model_name) if model_name else ""

        firmware_str = ""
        if firmware is not None:
            firmware_str = str(firmware).strip()

        # Специальная обработка для NEVASP111 (уже существующая)
        if 'NEVASP111' in model_name:
            if not firmware_str:
                return "METER_MODEL_NEVA_CP111_REV2"

            # Версии для REV2
            rev2_versions = [
                'ЛРВМ.SPN01.R.C.D.RU.0.0.2',
                'ЛРВМ.SPN01.R.C.D.RU.0.0.3',
                'ЛРВМ.SPN01.R.C.D.RU.0.0.4'
            ]

            # Версии для базовой CP111
            cp111_versions = [
                'ЛРВМ.SPN01.R.C.D.RU',
                'ЛРВМ.SPN01.R.C.D.RU.0.0.1'
            ]

            if firmware_str in rev2_versions:
                return "METER_MODEL_NEVA_CP111_REV2"
            elif firmware_str in cp111_versions:
                return "METER_MODEL_NEVA_CP111"
            else:
                return "METER_MODEL_NEVA_CP111"

        # Берилл СТЭ 31 (простая модель без зависимостей от прошивки)
        if 'Берилл СТЭ 31' in model_name:
            return "METER_MODEL_BERILL_STE_31"

        # СГБМ-4
        if 'СГБМ-4' in model_name:
            return "METER_MODEL_BETAR_SGBM_1_6M"

        # CE207 СПОДЕС
        if 'CE207 СПОДЭС' in model_name:
            if not firmware_str:
                return "METER_MODEL_CE207_SPODES_V10"

            # Версии для V10_8
            v10_8_versions = [
                'ЛРВМ.SPE8.R.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.8',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.8',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.8'
            ]

            # Версии для V10 (базовая)
            v10_versions = [
                'ЛРВМ.SPE.A.C.D.RU'
                'ЛРВМ.SPE.A.C.D.RU.0.0.1'
                'ЛРВМ.SPE.A.C.D.RU.0.0.2',
                'ЛРВМ.SPE.A.C.D.RU.0.0.3',
                'ЛРВМ.SPE.A.C.D.RU.0.0.4',
                'ЛРВМ.SPE.A.C.D.RU.0.0.5',
                'ЛРВМ.SPE.R.C.D.RU',
                'ЛРВМ.SPE.R.C.D.RU.0.0.1',
                'ЛРВМ.SPE.R.C.D.RU.0.0.2',
                'ЛРВМ.SPE.R.C.D.RU.0.0.3',
                'ЛРВМ.SPE.R.C.D.RU.0.0.4',
                'ЛРВМ.SPE.R.C.D.RU.0.0.5',
                'ЛРВМ.SPE.W.C.D.RU',
                'ЛРВМ.SPE.W.C.D.RU.0.0.1',
                'ЛРВМ.SPE.W.C.D.RU.0.0.2',
                'ЛРВМ.SPE.W.C.D.RU.0.0.3',
                'ЛРВМ.SPE.W.C.D.RU.0.0.4',
                'ЛРВМ.SPE.W.C.D.RU.0.0.5'
            ]

            if firmware_str in v10_8_versions:
                return "METER_MODEL_CE207_SPODES_V10_8"
            elif firmware_str in v10_versions or firmware_str == "или":
                return "METER_MODEL_CE207_SPODES_V10"
            else:
                return "METER_MODEL_CE207_SPODES_V10"

        # CE208 IEC (простая модель)
        if 'CE208 IEC' in model_name:
            return "METER_MODEL_CE208"

        # CE208 СПОДЕС
        if 'CE208 СПОДЕС' in model_name:
            if not firmware_str:
                return "METER_MODEL_CE208_SPODES_V10"

            # Базовые версии CE208_SPODES
            ce208_base = [
                'ЛРВМ.SPE.A.C.D.RU.0.0.1',
                'ЛРВМ.SPE.A.C.D.RU.0.0.2'
            ]

            # Версии для V10_8
            v10_8_versions = [
                'ЛРВМ.SPE8.R.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.8',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.8',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.8'
            ]

            # Версии для V10
            v10_versions = [
                'ЛРВМ.SPE.A.C.D.RU'
                'ЛРВМ.SPE.A.C.D.RU.0.0.1'
                'ЛРВМ.SPE.A.C.D.RU.0.0.2',
                'ЛРВМ.SPE.A.C.D.RU.0.0.3',
                'ЛРВМ.SPE.A.C.D.RU.0.0.4',
                'ЛРВМ.SPE.A.C.D.RU.0.0.5',
                'ЛРВМ.SPE.R.C.D.RU',
                'ЛРВМ.SPE.R.C.D.RU.0.0.1',
                'ЛРВМ.SPE.R.C.D.RU.0.0.2',
                'ЛРВМ.SPE.R.C.D.RU.0.0.3',
                'ЛРВМ.SPE.R.C.D.RU.0.0.4',
                'ЛРВМ.SPE.R.C.D.RU.0.0.5',
                'ЛРВМ.SPE.W.C.D.RU',
                'ЛРВМ.SPE.W.C.D.RU.0.0.1',
                'ЛРВМ.SPE.W.C.D.RU.0.0.2',
                'ЛРВМ.SPE.W.C.D.RU.0.0.3',
                'ЛРВМ.SPE.W.C.D.RU.0.0.4',
                'ЛРВМ.SPE.W.C.D.RU.0.0.5'
            ]

            if firmware_str in ce208_base:
                return "METER_MODEL_CE208_SPODES"
            elif firmware_str in v10_8_versions:
                return "METER_MODEL_CE208_SPODES_V10_8"
            elif firmware_str in v10_versions:
                return "METER_MODEL_CE208_SPODES_V10"
            else:
                return "METER_MODEL_CE208_SPODES_V10"

        # CE307 СПОДЕС
        if 'CE307 СПОДЕС' in model_name:
            if not firmware_str:
                return "METER_MODEL_CE307_SPODES_V10"

            # Версии для V10_8
            v10_8_versions = [
                'ЛРВМ.SPE8.R.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.8',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.8',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.8'
            ]

            # Версии для V10
            v10_versions = [
                'ЛРВМ.SPE.A.C.D.RU'
                'ЛРВМ.SPE.A.C.D.RU.0.0.1'
                'ЛРВМ.SPE.A.C.D.RU.0.0.2',
                'ЛРВМ.SPE.A.C.D.RU.0.0.3',
                'ЛРВМ.SPE.A.C.D.RU.0.0.4',
                'ЛРВМ.SPE.A.C.D.RU.0.0.5',
                'ЛРВМ.SPE.R.C.D.RU',
                'ЛРВМ.SPE.R.C.D.RU.0.0.1',
                'ЛРВМ.SPE.R.C.D.RU.0.0.2',
                'ЛРВМ.SPE.R.C.D.RU.0.0.3',
                'ЛРВМ.SPE.R.C.D.RU.0.0.4',
                'ЛРВМ.SPE.R.C.D.RU.0.0.5',
                'ЛРВМ.SPE.W.C.D.RU',
                'ЛРВМ.SPE.W.C.D.RU.0.0.1',
                'ЛРВМ.SPE.W.C.D.RU.0.0.2',
                'ЛРВМ.SPE.W.C.D.RU.0.0.3',
                'ЛРВМ.SPE.W.C.D.RU.0.0.4',
                'ЛРВМ.SPE.W.C.D.RU.0.0.5'
            ]

            if firmware_str in v10_8_versions:
                return "METER_MODEL_CE307_SPODES_V10_8"
            elif firmware_str in v10_versions:
                return "METER_MODEL_CE307_SPODES_V10"
            else:
                return "METER_MODEL_CE307_SPODES_V10"

        # CE308 IEC (простая модель)
        if 'CE308 IEC' in model_name:
            return "METER_MODEL_CE308"

        # CE308 СПОДЕС
        if 'CE308 СПОДЕС' in model_name:
            if not firmware_str:
                return "METER_MODEL_CE308_SPODES_V10"

            # Базовые версии CE308_SPODES
            ce308_base = [
                'ЛРВМ.SPE.A.C.D.RU',
                'ЛРВМ.SPE.A.C.D.RU.0.0.1'
            ]

            # Версии для V10_8
            v10_8_versions = [
                'ЛРВМ.SPE8.R.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.R.C.D.RU.0.0.8',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.W.C.D.RU.0.0.8',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.2',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.3',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.4',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.5',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.6',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.7',
                'ЛРВМ.SPE8.A.C.D.RU.0.0.8'
            ]

            # Версии для V10
            v10_versions = [
                'ЛРВМ.SPE.A.C.D.RU'
                'ЛРВМ.SPE.A.C.D.RU.0.0.1'
                'ЛРВМ.SPE.A.C.D.RU.0.0.2',
                'ЛРВМ.SPE.A.C.D.RU.0.0.3',
                'ЛРВМ.SPE.A.C.D.RU.0.0.4',
                'ЛРВМ.SPE.A.C.D.RU.0.0.5',
                'ЛРВМ.SPE.R.C.D.RU',
                'ЛРВМ.SPE.R.C.D.RU.0.0.1',
                'ЛРВМ.SPE.R.C.D.RU.0.0.2',
                'ЛРВМ.SPE.R.C.D.RU.0.0.3',
                'ЛРВМ.SPE.R.C.D.RU.0.0.4',
                'ЛРВМ.SPE.R.C.D.RU.0.0.5',
                'ЛРВМ.SPE.W.C.D.RU',
                'ЛРВМ.SPE.W.C.D.RU.0.0.1',
                'ЛРВМ.SPE.W.C.D.RU.0.0.2',
                'ЛРВМ.SPE.W.C.D.RU.0.0.3',
                'ЛРВМ.SPE.W.C.D.RU.0.0.4',
                'ЛРВМ.SPE.W.C.D.RU.0.0.5'
            ]

            if firmware_str in ce308_base:
                return "METER_MODEL_CE308_SPODES"
            elif firmware_str in v10_versions:
                return "METER_MODEL_CE308_SPODES_V10"
            elif firmware_str in v10_8_versions:
                return "METER_MODEL_CE308_SPODES_V10_8"
            else:
                return "METER_MODEL_CE308_SPODES_V10"

        # Chronos
        if 'Chronos' in model_name:
            return "METER_MODEL_CHRONOS"

        # УРТ-100
        if 'УРТ-100' in model_name:
            return "METER_MODEL_INDIVID_V2"

        # Индивид Саяны
        if 'Индивид Саяны' in model_name:
            return "METER_MODEL_INDIVID_V3_PLUS"

        # Каскад-11-С1
        if 'Каскад-11-С1' in model_name:
            return "METER_MODEL_KASKAD_11_C1"

        # ЛЭ 221.1.RF.D0
        if 'ЛЭ 221.1.RF.D0' in model_name:
            return "METER_MODEL_LE_221"

        # Меркурий 150 СПОДЕС
        if 'Меркурий 150 СПОДЕС' in model_name:
            return "METER_MODEL_MERCURY_150_V1_70"

        # Меркурий 204 (без СПОДЕС)
        if 'Меркурий 204' in model_name and 'СПОДЕС' not in model_name:
            return "METER_MODEL_MERCURY_204"

        # Меркурий 204 СПОДЕС
        if 'Меркурий 204 СПОДЕС' in model_name:
            if not firmware_str:
                return "METER_MODEL_MERCURY_204D"

            # Обработка разных версий прошивок для Меркурий 204 СПОДЕС
            mercury_204_versions = {
                'v68': ['ЛРВМ.GSPD.S.C.D.RU', 'ЛРВМ.GSPD.S.C.D.RU.0.0.1', 'ЛРВМ.SPD.S.C.D.RU.0.0.1',
                        'ЛРВМ.SPD.S.C.D.RU',
                        'ЛРВМ.SPD.S.C.D.RU.0.0.2',
                        'ЛРВМ.SPD.S.C.D.RU.0.0.3',
                        'ЛРВМ.SPD.S.C.D.RU.0.0.4',
                        'ЛРВМ.SPD.S.C.D.RU.0.0.5',
                        'ЛРВМ.SPD.S.C.D.RU.0.0.6',
                        'ЛРВМ.SPD.S.C.D.RU.0.0.7',
                        'ЛРВМ.SPD.S.C.D.RU.0.0.8',
                        'ЛРВМ.SPD.S.C.D.RU.0.0.9'],
                'v69': ['ЛРВМ.SPD.S.C.D.RU.0.0.12'],
                'v0_19': ['ЛРВМ.SPDXV19.R.C.D.RU.0.0.1'],
                'x': ['ЛРВМ.SPDX.S.C.D.RU.0.0.1', 'ЛРВМ.SPDX.R.C.D.RU',
                      'ЛРВМ.SPDX.R.C.D.RU.0.0.2', 'ЛРВМ.SPDX.R.C.D.RU.0.0.3',
                      'ЛРВМ.SPDX.R.C.D.RU.0.0.4'],
                'v1_56': ['ЛРВМ.SPDX.R.C.D.RU.0.0.6', 'ЛРВМ.SPDX.R.C.D.RU.0.0.7',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.8', 'ЛРВМ.SPDX.R.C.D.RU.0.0.9',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.10', 'ЛРВМ.SPDX.R.C.D.RU.0.0.11']
            }

            for version_type, versions in mercury_204_versions.items():
                if firmware_str in versions:
                    if version_type == 'v68':
                        return "METER_MODEL_MERCURY_204D_V68"
                    elif version_type == 'v69':
                        return "METER_MODEL_MERCURY_204D_V69"
                    elif version_type == 'x':
                        return "METER_MODEL_MERCURY_204X"
                    elif version_type == 'v0_19':
                        return "METER_MODEL_MERCURY_204X_V0_19"
                    elif version_type == 'v1_56':
                        return "METER_MODEL_MERCURY_204X_V1_56"

            # Если не нашли специфичную версию, возвращаем базовую
            return "METER_MODEL_MERCURY_204D"

        # Меркурий 206
        if 'Меркурий 206' in model_name:
            return "METER_MODEL_MERCURY_206"

        # Меркурий 208 (без СПОДЕС)
        if 'Меркурий 208' in model_name and 'СПОДЕС' not in model_name:
            return "METER_MODEL_MERCURY_208"

        # Меркурий 208 СПОДЕС
        if 'Меркурий 208 СПОДЕС' in model_name:
            if not firmware_str:
                return "METER_MODEL_MERCURY_208D"

            # Обработка разных версий прошивок для Меркурий 208 СПОДЕС
            mercury_208_versions = {
                'base': ['ЛРВМ.SPD.S.C.D.RU.0.0.1', 'ЛРВМ.SPD.S.C.D.RU',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.2',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.3',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.4',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.5',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.6',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.7',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.8',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.9'],
                'v68': ['ЛРВМ.SPD.S.C.D.RU.0.0.11'],
                'v69': ['ЛРВМ.SPD.S.C.D.RU.0.0.12'],
                'x': ['ЛРВМ.SPDX.R.C.D.RU',
                      'ЛРВМ.SPDX.R.C.D.RU.0.0.1',
                      'ЛРВМ.SPDX.R.C.D.RU.0.0.2'],
                'v0_19': ['ЛРВМ.SPDX.R.C.D.RU.0.0.3', 'ЛРВМ.SPDX.R.C.D.RU.0.0.4',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.5'],
                'v1_57': ['ЛРВМ.INCGSM.R.C.D.RU.0.0.4', 'ЛРВМ.SPDX.R.C.D.RU.0.0.6',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.7', 'ЛРВМ.SPDX.R.C.D.RU.0.0.8',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.9', 'ЛРВМ.SPDX.R.C.D.RU.0.0.10',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.11', 'ЛРВМ.SPDX.R.C.D.RU.0.0.12']
            }

            for version_type, versions in mercury_208_versions.items():
                if firmware_str in versions:
                    if version_type == 'base':
                        return "METER_MODEL_MERCURY_208D"
                    elif version_type == 'v68':
                        return "METER_MODEL_MERCURY_208D_V68"
                    elif version_type == 'v69':
                        return "METER_MODEL_MERCURY_208D_V69"
                    elif version_type == 'x':
                        return "METER_MODEL_MERCURY_208X"
                    elif version_type == 'v0_19':
                        return "METER_MODEL_MERCURY_208X_V0_19"
                    elif version_type == 'v1_57':
                        return "METER_MODEL_MERCURY_208X_V1_57"

            # Если не нашли специфичную версию, возвращаем базовую
            return "METER_MODEL_MERCURY_208D"

        # Меркурий 234 (без СПОДЕС)
        if 'Меркурий 234' in model_name and 'СПОДЕС' not in model_name:
            return "METER_MODEL_MERCURY_234"

        # Меркурий 234 СПОДЕС
        if 'Меркурий 234 СПОДЕС' in model_name:
            if not firmware_str:
                return "METER_MODEL_MERCURY_234D"

            # Обработка разных версий прошивок для Меркурий 234 СПОДЕС
            mercury_234_versions = {
                'base': ['ЛРВМ.SPD.S.C.D.RU.0.0.1', 'ЛРВМ.SPD.S.C.D.RU',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.2',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.3',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.4',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.5',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.6',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.7',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.8',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.9', 'ЛРВМ.SPD.A.C.D.RU.0.0.1', 'ЛРВМ.SPD.A.C.D.RU.0.0.2',
                         'ЛРВМ.SPD.A.C.D.RU.0.0.3', 'ЛРВМ.SPD.A.C.D.RU.0.0.4', 'ЛРВМ.SPD.A.C.D.RU.0.0.5',
                         'ЛРВМ.SPD.A.C.D.RU'],
                'v68': ['ЛРВМ.SPD.A.C.D.RU.0.0.6'],
                'v69': ['ЛРВМ.SPD.A.C.D.RU.0.0.7'],
                'mx2': ['ЛРВМ.SPDX.R.C.D.RU', 'ЛРВМ.SPDX.R.C.D.RU.0.0.1',
                        'ЛРВМ.SPDX.R.C.D.RU.0.0.2', 'ЛРВМ.SPDX.R.C.D.RU.0.0.3',
                        'ЛРВМ.SPDX.W.C.D.RU', 'ЛРВМ.SPDX.W.C.D.RU.0.0.1',
                        'ЛРВМ.SPDX.W.C.D.RU.0.0.2', 'ЛРВМ.SPDX.W.C.D.RU.0.0.3'],
                'v1_56': ['ЛРВМ.INCGSM.R.C.D.RU.0.0.2', 'ЛРВМ.INCGSM.R.C.D.RU.0.0.3',
                          'ЛРВМ.INCGSM.R.C.D.RU.0.0.4', 'ЛРВМ.SPDX.R.C.D.RU.0.0.6', 'ЛРВМ.SPDX.R.C.D.RU.0.0.7',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.8', 'ЛРВМ.SPDX.R.C.D.RU.0.0.9', 'ЛРВМ.SPDX.R.C.D.RU.0.0.10'
                                                                                  'ЛРВМ.SPDX.R.C.D.RU.0.0.11',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.12',
                          'ЛРВМ.SPDX.W.C.D.RU.0.0.4', 'ЛРВМ.SPDX.W.C.D.RU.0.0.5', 'ЛРВМ.SPDX.W.C.D.RU.0.0.6']
            }

            for version_type, versions in mercury_234_versions.items():
                if firmware_str in versions:
                    if version_type == 'base':
                        return "METER_MODEL_MERCURY_234D"
                    elif version_type == 'v68':
                        return "METER_MODEL_MERCURY_234D_V68"
                    elif version_type == 'v69':
                        return "METER_MODEL_MERCURY_234D_V69"
                    elif version_type == 'mx2':
                        return "METER_MODEL_MERCURY_234MX2"
                    elif version_type == 'v1_56':
                        return "METER_MODEL_MERCURY_234X_V1_56"

            # Если не нашли специфичную версию, возвращаем базовую
            return "METER_MODEL_MERCURY_234D"

        # Меркурий 238 (без СПОДЕС)
        if 'Меркурий 238' in model_name and 'СПОДЭС' not in model_name:
            return "METER_MODEL_MERCURY_238"

        # Меркурий 238 СПОДЕС
        if 'Меркурий 238 СПОДЭС' in model_name:
            if not firmware_str:
                return "METER_MODEL_MERCURY_238D"

            # Обработка разных версий прошивок для Меркурий 238 СПОДЕС
            mercury_238_versions = {
                'base': ['ЛРВМ.SPD.S.C.D.RU.0.0.1', 'ЛРВМ.SPD.S.C.D.RU',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.2',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.3',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.4',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.5',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.6',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.7',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.8',
                         'ЛРВМ.SPD.S.C.D.RU.0.0.9', 'ЛРВМ.SPD.A.C.D.RU', 'ЛРВМ.SPD.A.C.D.RU.0.0.1',
                         'ЛРВМ.SPD.A.C.D.RU.0.0.2', 'ЛРВМ.SPD.A.C.D.RU.0.0.3', 'ЛРВМ.SPD.A.C.D.RU.0.0.4',
                         'ЛРВМ.SPD.A.C.D.RU.0.0.5'],
                'v68': ['ЛРВМ.GSPD.S.C.D.RU.0.0.1'],
                'v69': ['ЛРВМ.SPD.S.C.D.RU.0.0.12', 'ЛРВМ.SPD.S.C.D.RU.0.0.11'],
                'x': ['ЛРВМ.SPDX.R.C.D.RU', 'ЛРВМ.SPDX.R.C.D.RU.0.0.1',
                      'ЛРВМ.SPDX.R.C.D.RU.0.0.2', 'ЛРВМ.SPDX.R.C.D.RU.0.0.3',
                      'ЛРВМ.SPDX.W.C.D.RU', 'ЛРВМ.SPDX.W.C.D.RU.0.0.1',
                      'ЛРВМ.SPDX.W.C.D.RU.0.0.2', 'ЛРВМ.SPDX.W.C.D.RU.0.0.3'],
                'v1_57': ['ЛРВМ.INCGSM.R.C.D.RU.0.0.2', 'ЛРВМ.INCGSM.R.C.D.RU.0.0.3',
                          'ЛРВМ.INCGSM.R.C.D.RU.0.0.4', 'ЛРВМ.SPDX.R.C.D.RU.0.0.6',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.7', 'ЛРВМ.SPDX.R.C.D.RU.0.0.8',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.9', 'ЛРВМ.SPDX.R.C.D.RU.0.0.10',
                          'ЛРВМ.SPDX.R.C.D.RU.0.0.11', 'ЛРВМ.SPDX.R.C.D.RU.0.0.12']
            }

            for version_type, versions in mercury_238_versions.items():
                if firmware_str in versions:
                    if version_type == 'base':
                        return "METER_MODEL_MERCURY_238D"
                    elif version_type == 'v68':
                        return "METER_MODEL_MERCURY_238D_V68"
                    elif version_type == 'v69':
                        return "METER_MODEL_MERCURY_238D_V69"
                    elif version_type == 'x':
                        return "METER_MODEL_MERCURY_238X"
                    elif version_type == 'v1_57':
                        return "METER_MODEL_MERCURY_238X_V1_57"

            # Если не нашли специфичную версию, возвращаем базовую
            return "METER_MODEL_MERCURY_238D"

        # Мир 04, 05, 07
        if 'Мир 04' in model_name:
            return "METER_MODEL_MIR_C_04"
        if 'Мир 05' in model_name:
            return "METER_MODEL_MIR_C_05"
        if 'Мир 07' in model_name:
            return "METER_MODEL_MIR_C_07"

        # Милур-107S
        if 'Милур-107S' in model_name:
            if not firmware_str:
                return "METER_MODEL_MLR_107S"

            base_versions = ['ЛРВМ.MLR.W.C.D.RU.0.0.1', 'ЛРВМ.MLR.W.C.D.RU.0.0.2',
                             'ЛРВМ.MLR.W.C.D.RU.0.0.3', '']

            if firmware_str in base_versions:
                return "METER_MODEL_MLR_107S"
            elif firmware_str == 'ЛРВМ.MLR.W.C.D.RU.0.0.4':
                return "METER_MODEL_MLR_107S_V1_18_0_107"
            else:
                return "METER_MODEL_MLR_107S"

        # Милур-307S
        if 'Милур-307S' in model_name:
            if not firmware_str:
                return "METER_MODEL_MLR_307S"

            base_versions = ['ЛРВМ.MLR.W.C.D.RU.0.0.1', 'ЛРВМ.MLR.W.C.D.RU.0.0.2',
                             'ЛРВМ.MLR.W.C.D.RU.0.0.3', '']

            if firmware_str in base_versions:
                return "METER_MODEL_MLR_307S"
            elif firmware_str == 'ЛРВМ.MLR.W.C.D.RU.0.0.4':
                return "METER_MODEL_MLR_307S_V1_18_0_107"
            else:
                return "METER_MODEL_MLR_307S"

        # Нартис-100 и Нартис-300
        if 'Нартис-100' in model_name:
            if firmware_str == 'ЛРВМ.NRTHW2.W.C.D.RU.0.0.2':
                return "METER_MODEL_NARTIS_I300_W113-2"
            return "METER_MODEL_NARTIS_100"

        if 'Нартис-300' in model_name:
            if firmware_str == 'ЛРВМ.NRTHW2.W.C.D.RU.0.0.1':
                return "METER_MODEL_NARTIS_I300_W113-2"
            return "METER_MODEL_NARTIS_300"

        # NEVACT414 и NEVACT414v.2
        if 'NEVACT414' in model_name:
            print(f"DEBUG: Обработка NEVACT414, модель: '{model_name}', прошивка: '{firmware_str}'")

            # Проверяем версию v.2
            if 'NEVACT414v.2' in model_name or 'NEVACT414v2' in model_name or 'NEVACT414 v.2' in model_name:
                # Для NEVACT414v.2
                if firmware_str in ['ЛРВМ.SPN03.W.C.D.RU.0.0.2']:
                    return "METER_MODEL_NEVA_CT414_V2"
                else:
                    # Если прошивка не указана или другая, возвращаем базовую v2
                    return "METER_MODEL_NEVA_CT414_V2"
            else:
                # Для базовой NEVACT414
                if firmware_str in ['ЛРВМ.SPN.W.C.D.RU.0.0.5', 'ЛРВМ.SPN.W.C.D.RU.0.0.6']:
                    return "METER_MODEL_NEVA_CT414"
                else:
                    # Если прошивка не указана или другая, возвращаем базовую
                    return "METER_MODEL_NEVA_CT414"

        # Остальные модели без зависимостей от прошивки
        model_mapping = {
            'Нартис-302': 'METER_MODEL_NARTIS_302',
            'Нартис-102': 'METER_MODEL_NARTIS_102',
            'Диммер Meter': 'METER_MODEL_PREM',
            'Аква Пульс': 'METER_MODEL_PULS',
            'ПУЛЬС СТК': 'METER_MODEL_PULS_STK',
            'Рокип102.03': 'METER_MODEL_ROK_102',
            'Рокип103.03': 'METER_MODEL_ROK_103_03',
            'Рокип103.06': 'METER_MODEL_ROK_103_06',
            'Вега Си Вода': 'METER_MODEL_SI_WATER',
            'СВК-15-3-2': 'METER_MODEL_SVK',
            'Topenar Compact': 'METER_MODEL_TOPENAR_COMPACT',
            'Для двух приборов учета': 'METER_MODEL_TWIN_COMPLETE',
            'Вектор 100': 'METER_MODEL_VECTOR_100',
            'Вектор101': 'METER_MODEL_VECTOR_101',
            'Вектор 300': 'METER_MODEL_VECTOR_300',
            'WFK20': 'METER_MODEL_WFK_20',
            'WFW20': 'METER_MODEL_WFW_20',
            'ЦЭ2726А A1.S.RF.R03.Z.M': 'METER_MODEL_ZIP_2726A',
            'ЦЭ2727А S.RF.OP.B04.Z.R': 'METER_MODEL_ZIP_2727A',
            'ЦЭ2726А A1.S.RF.OP.W03.Z.R': 'METER_MODEL_ZIP_2727A',
            'ЦЭ2727А S.RF.R02': 'METER_MODEL_ZIP_2727A',
            'ЦЭ2727А T.RF.OP.B04 5-10A': 'METER_MODEL_ZIP_2727A',
            'ОЭС 1-Ф': 'METER_MODEL_OES_1',
            'ОЭС 3-Ф': 'METER_MODEL_OES_3',
            'NEVAMT115': 'METER_MODEL_NEVA_MT115',
            'NEVACT221': 'METER_MODEL_NEVA_CT221',
            'NEVACT413': 'METER_MODEL_NEVA_CT413',
            'NEVASP311': 'METER_MODEL_NEVA_SP311'
        }

        # Поиск в базовом маппинге
        for key, value in model_mapping.items():
            if key in model_name:
                return value

        # Если не нашли, возвращаем преобразованное название
        clean_name = model_name.replace(' ', '_').replace('-', '_').replace('.', '_')
        return f"METER_MODEL_{clean_name}"

    def closeEvent(self, event):
        """Обработка закрытия окна"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.terminate()
            self.processing_thread.wait()
        event.accept()


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        # Устанавливаем ограничение на рекурсию
        sys.setrecursionlimit(10000)

        viewer = CSVViewer()
        viewer.show()

        sys.exit(app.exec_())

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)