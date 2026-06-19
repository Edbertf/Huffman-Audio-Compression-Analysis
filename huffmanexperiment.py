import numpy as np
import pandas as pd
import heapq
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

#Generating 5 DIfferent Signals
def generate_signals(fs=8000, duration=2.0, frequency=440):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    rng = np.random.default_rng(42)

    signals = {
        "Silence / Low Variation": np.zeros_like(t),

        "Sine Wave": np.sin(2 * np.pi * frequency * t),

        "Square Wave": np.where(np.sin(2 * np.pi * frequency * t) >= 0,1,-1),

        "Sawtooth Wave": (2 * (frequency * t - np.floor(0.5 + frequency * t))),

        "White Noise": rng.uniform(-1, 1, len(t))
    }
    return signals, t


#Converting Signals to Byte
def convert_to_byte_values(signal):
    clipped_signal = np.clip(signal, -1, 1)
    byte_values = np.round((clipped_signal + 1) * 127.5).astype(np.uint8)
    return byte_values


#Building Huffman Tree
def build_huffman_tree(frequencies):
    counter = itertools.count()
    heap = []

    for symbol, frequency in frequencies.items():
        heapq.heappush(heap, (frequency, next(counter), symbol))

    if len(heap) == 1:
        return heap[0][2]

    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)

        merged_node = (node1, node2)
        merged_frequency = freq1 + freq2

        heapq.heappush(heap, (merged_frequency, next(counter), merged_node))

    return heap[0][2]


#Generating Huffman Codes
def generate_huffman_codes(tree):
    codes = {}

    def traverse(node, current_code):
        if not isinstance(node, tuple):
            codes[node] = current_code if current_code != "" else "0"
            return

        left, right = node
        traverse(left, current_code + "0")
        traverse(right, current_code + "1")

    traverse(tree, "")
    return codes


def huffman_coding(byte_data):
    data_list = [int(x) for x in byte_data]
    frequencies = Counter(data_list)

    tree = build_huffman_tree(frequencies)
    codes = generate_huffman_codes(tree)

    return frequencies, codes


#Calculating Compression Metrics
def calculate_compression_metrics(byte_data):
    frequencies, codes = huffman_coding(byte_data)

    number_of_symbols = len(byte_data)
    unique_symbols = len(frequencies)

    original_size = number_of_symbols * 8

    huffman_encoded_size = sum(
        frequencies[symbol] * len(codes[symbol])
        for symbol in frequencies
    )

    average_code_length = huffman_encoded_size / number_of_symbols
    compression_ratio = huffman_encoded_size / original_size
    space_saving = (1 - compression_ratio) * 100

    metrics = {
        "Unique Symbols": unique_symbols,
        "Average Code Length": average_code_length,
        "Original Size (bits)": original_size,
        "Huffman Encoded Size (bits)": huffman_encoded_size,
        "Compression Ratio": compression_ratio,
        "Space Saving (%)": space_saving
    }

    return metrics, frequencies, codes


#Visualization
def make_safe_filename(signal_name):
    return signal_name.replace("/", "-").replace(" ", "_")


def plot_waveform(signal_name, signal, t, output_folder):
    plt.figure(figsize=(8, 3))

    # Only the first 500 samples are plotted so the waveform shape is visible.
    plt.plot(t[:500], signal[:500])

    plt.title(f"Waveform of {signal_name}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    safe_name = make_safe_filename(signal_name)
    file_path = output_folder / f"{safe_name}_waveform.png"

    plt.savefig(file_path, dpi=300)
    plt.close()


def plot_symbol_distribution(signal_name, frequencies, output_folder):
    symbols = sorted(frequencies.keys())
    counts = [frequencies[symbol] for symbol in symbols]

    plt.figure(figsize=(8, 3))
    plt.bar(symbols, counts, width=0.8)

    plt.title(f"Symbol Frequency Distribution of {signal_name}")
    plt.xlabel("Byte Value")
    plt.ylabel("Frequency")

    plt.xlim(-5, 260)
    plt.xticks([0, 64, 128, 192, 255])

    plt.tight_layout()

    safe_name = make_safe_filename(signal_name)
    file_path = output_folder / f"{safe_name}_distribution.png"

    plt.savefig(file_path, dpi=300)
    plt.close()

def save_results_table_as_image(results_df, output_folder):
    display_df = results_df.copy()

    display_df["Average Code Length"] = display_df["Average Code Length"].map(
        lambda x: f"{x:.3f}"
    )
    display_df["Compression Ratio"] = display_df["Compression Ratio"].map(
        lambda x: f"{x:.3f}"
    )
    display_df["Space Saving (%)"] = display_df["Space Saving (%)"].map(
        lambda x: f"{x:.2f}"
    )

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    plt.tight_layout()

    image_path = output_folder / "compression_results_table.png"
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()

    return image_path


#Output
def print_results_pretty(results_df):
    print("\n=== Huffman Coding Compression Results ===\n")

    for _, row in results_df.iterrows():
        print(f"Signal Type                  : {row['Signal Type']}")
        print(f"Unique Symbols               : {row['Unique Symbols']}")
        print(f"Average Code Length          : {row['Average Code Length']:.6f}")
        print(f"Original Size (bits)         : {row['Original Size (bits)']}")
        print(f"Huffman Encoded Size (bits)  : {row['Huffman Encoded Size (bits)']}")
        print(f"Compression Ratio            : {row['Compression Ratio']:.6f}")
        print(f"Space Saving (%)             : {row['Space Saving (%)']:.6f}")
        print("-" * 55)


def save_codebook_csv(codebook_summary, output_folder):
    codebook_df = pd.DataFrame(codebook_summary)

    codebook_path = output_folder / "huffman_codebooks.csv"

    try:
        codebook_df.to_csv(codebook_path, index=False)
    except PermissionError:
        codebook_path = output_folder / "huffman_codebooks_new.csv"
        codebook_df.to_csv(codebook_path, index=False)

    return codebook_path


#Main
def run_experiment():
    fs = 8000
    duration = 2.0
    frequency = 440

    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()

    output_folder = base_dir / "output"
    output_folder.mkdir(parents=True, exist_ok=True)

    signals, t = generate_signals(
        fs=fs,
        duration=duration,
        frequency=frequency
    )

    results = []
    codebook_summary = []

    for signal_name, signal in signals.items():
        byte_data = convert_to_byte_values(signal)

        metrics, frequencies, codes = calculate_compression_metrics(byte_data)

        metrics["Signal Type"] = signal_name
        results.append(metrics)

        for symbol in sorted(frequencies.keys()):
            codebook_summary.append({
                "Signal Type": signal_name,
                "Symbol": symbol,
                "Frequency": frequencies[symbol],
                "Huffman Code": codes[symbol],
                "Code Length": len(codes[symbol])
            })

        plot_waveform(signal_name, signal, t, output_folder)
        plot_symbol_distribution(signal_name, frequencies, output_folder)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Signal Type",
            "Unique Symbols",
            "Average Code Length",
            "Original Size (bits)",
            "Huffman Encoded Size (bits)",
            "Compression Ratio",
            "Space Saving (%)"
        ]
    ]

    print_results_pretty(results_df)

    result_image_path = save_results_table_as_image(results_df, output_folder)
    codebook_path = save_codebook_csv(codebook_summary, output_folder)

    print("\nFiles saved successfully.")
    print(f"Output folder       : {output_folder}")
    print(f"Results table image : {result_image_path}")
    print(f"Huffman codebook CSV: {codebook_path}")


if __name__ == "__main__":
    run_experiment()