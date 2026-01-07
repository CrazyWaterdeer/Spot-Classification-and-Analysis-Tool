"""
Script to measure import times for SCAT modules.
"""
import time

def measure_import(name, import_func):
    start = time.perf_counter()
    try:
        import_func()
        elapsed = time.perf_counter() - start
        print(f"{name:40} {elapsed*1000:8.1f} ms")
        return elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"{name:40} {elapsed*1000:8.1f} ms (FAILED: {e})")
        return elapsed

print("=" * 55)
print("Import Time Analysis for SCAT")
print("=" * 55)

total = 0

# Standard library (baseline)
total += measure_import("sys", lambda: __import__('sys'))
total += measure_import("os", lambda: __import__('os'))
total += measure_import("pathlib", lambda: __import__('pathlib'))

print("-" * 55)

# Heavy libraries
total += measure_import("numpy", lambda: __import__('numpy'))
total += measure_import("pandas", lambda: __import__('pandas'))
total += measure_import("cv2 (OpenCV)", lambda: __import__('cv2'))
total += measure_import("PIL", lambda: __import__('PIL'))

print("-" * 55)

# sklearn (the heavy one)
total += measure_import("sklearn.ensemble", lambda: __import__('sklearn.ensemble'))
total += measure_import("sklearn.model_selection", lambda: __import__('sklearn.model_selection'))
total += measure_import("sklearn.metrics", lambda: __import__('sklearn.metrics'))
total += measure_import("sklearn.preprocessing", lambda: __import__('sklearn.preprocessing'))
total += measure_import("sklearn.decomposition", lambda: __import__('sklearn.decomposition'))
total += measure_import("sklearn.cluster", lambda: __import__('sklearn.cluster'))

print("-" * 55)

# Visualization
total += measure_import("matplotlib", lambda: __import__('matplotlib'))
total += measure_import("matplotlib.pyplot", lambda: __import__('matplotlib.pyplot'))
total += measure_import("seaborn", lambda: __import__('seaborn'))

print("-" * 55)

# Qt (PySide6)
total += measure_import("PySide6.QtWidgets", lambda: __import__('PySide6.QtWidgets'))
total += measure_import("PySide6.QtCore", lambda: __import__('PySide6.QtCore'))
total += measure_import("PySide6.QtGui", lambda: __import__('PySide6.QtGui'))

print("=" * 55)
print(f"{'TOTAL':40} {total*1000:8.1f} ms")
print("=" * 55)

print("\n** Analysis **")
print("- sklearn imports are typically the slowest (500-1500ms)")
print("- matplotlib + seaborn add 300-600ms")
print("- PySide6 adds 200-500ms")
print("- numpy/pandas/cv2 add 200-400ms each")
print("\nRecommendation: Use lazy imports for sklearn and matplotlib")
