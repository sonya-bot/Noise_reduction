import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os
import math # ダッシュボードの行数計算用に追加

# 音声信号からFFT振幅スペクトル(dB)を計算する関数 (変更なし)
def calculate_audio_stft(y, sr, n_fft=2048, hop_length=512):
    """
    音声信号からSTFT振幅スペクトル(dB)を計算
    """
    # トレンド除去 (DCオフセットの除去)
    detrended_signal = signal.detrend(y, type='constant')
    
    # STFTの計算
    stft = librosa.stft(detrended_signal, n_fft=n_fft, hop_length=hop_length)

    # 周波数軸の計算
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # 振幅スペクトルをdBに変換
    dbs = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    return dbs, hop_length

# 1つのAxes(グラフエリア)に対してスペクトルを描画する共通関数 (追加)
def plot_stft_on_ax(ax, hop_length, dbs, sr, title, show_xlabel=True, show_ylabel=True):
    img = librosa.display.specshow(dbs, sr=sr, hop_length=hop_length, 
                                    x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    
    # デザイン調整
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_ylim(0, 2000) # 風切り音の特性(0〜500Hz)を確認するため

    # 軸ラベル
    if show_ylabel:
        ax.set_ylabel('Frequency [Hz]', fontsize=12)
    if show_xlabel:
        ax.set_xlabel('Time [s]', fontsize=12)
    else:
        ax.set_xlabel('')

    return img

# ダッシュボード描画
def draw_dashboard(audio_files, input_path, output_path, fig_size, auto_save):
    n_plots = len(audio_files)
    if n_plots == 0:
        return

    n_cols = 3 # 見やすさのため3列に設定 (ファイル数に応じて調整可)
    n_rows = math.ceil(n_plots / n_cols)
    
    # 全体サイズの計算
    total_width = fig_size[0] * n_cols
    total_height = fig_size[1] * n_rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, total_height), layout='constrained')
    
    # ファイル数が1つの場合など、axesが配列にならないケースの対策
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    print("\n - ダッシュボードグラフ作成")
    img = None

    for i in range(n_cols * n_rows):
        ax = axes[i]
        if i < n_plots:
            file_name = audio_files[i]
            file_path = os.path.join(input_path, file_name)
            print(f"   - Processing {file_name}")
            
            y, sr = librosa.load(file_path, sr=None)
            dbs, hop_length = calculate_audio_stft(y, sr)
            
            # 軸ラベルの制御 (左端と下端のみ表示してスッキリさせる)
            show_y = (i % n_cols == 0)
            show_x = (i >= n_cols * (n_rows - 1))
            
            img = plot_stft_on_ax(ax, hop_length, dbs, sr, file_name, show_x, show_y)
        else:
            ax.axis('off') # 余ったグラフ領域は非表示にする

    if img is not None:
        cbar = fig.colorbar(img, ax=axes.ravel().tolist(), format="%+2.0f dB", pad=0.02)
        cbar.set_label('Amplitude [dB]')

    plt.suptitle("STFT Spectrum", fontsize=18)
    # plt.tight_layout()
    # plt.subplots_adjust(wspace=0.3, hspace=0.5)

    if auto_save:
        save_path = os.path.join(output_path, "Visualize_STFT.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ダッシュボード保存完了: {save_path}")
        plt.close()
    else:
        plt.show()

# 個別のグラフ作成 (既存機能を改修)
def visualize_fft(file_path, output_path, fig_size, auto_save):
    if not os.path.exists(file_path):
        print(f"エラー: '{file_path}' が見つかりません。")
        return

    y, sr = librosa.load(file_path, sr=None)
    print(f"Processing {os.path.basename(file_path)} (サンプリングレート: {sr} Hz, 長さ: {len(y)/sr:.2f} 秒)")
    
    S_db, hop_length = calculate_audio_stft(y, sr)

    fig, ax = plt.subplots(figsize=fig_size)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    img = plot_stft_on_ax(ax, hop_length, S_db, sr, f"STFT Spectrogram: {base_name}", show_xlabel=True, show_ylabel=True)
    
    # 個別グラフ用のカラーバー
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    
    plt.tight_layout()
    
    if auto_save:
        output_img = os.path.join(output_path, f"{base_name}_stft.png")
        plt.savefig(output_img, dpi=300, bbox_inches='tight')
        print(f"個別保存完了: {output_img}")
        plt.close()
    else:
        plt.show()

# メイン処理
if __name__ == "__main__":
    # フォルダパスを入力
    INPUT_PATH = "/Users/Souma/Develop/exp_data/recording_data/processed_data/scene_4"
    OUTPUT_PATH = "/Users/Souma/Develop/exp_data/recording_data/"

    # パラメータ設定
    FIG_SIZE = (6, 4)
    AUTO_SAVE = True
    
    if not os.path.isdir(INPUT_PATH):
        print(f"エラー: '{INPUT_PATH}' は有効なフォルダパスではありません。")
        exit(1)
    
    # フォルダ内の音声ファイルをリストアップ (拡張子でフィルタリング)
    audio_extensions = ['.wav']
    audio_files = [f for f in os.listdir(INPUT_PATH) if os.path.splitext(f)[1].lower() in audio_extensions]
    
    # ファイル名でソート（ダッシュボードの配置順を揃えるため）
    audio_files.sort()
    
    if not audio_files:
        print(f"フォルダ '{INPUT_PATH}' に音声ファイルが見つかりません。")
        exit(1)
    
    print(f"見つかった音声ファイル: {audio_files}")
    
    # ダッシュボードの作成
    draw_dashboard(audio_files, INPUT_PATH, OUTPUT_PATH, FIG_SIZE, AUTO_SAVE)
    
    # 個別ファイルの作成
    # print("\n - 個別グラフ作成")
    # for file_name in audio_files:
    #     file_path = os.path.join(INPUT_PATH, file_name)
    #     visualize_fft(file_path, OUTPUT_PATH, FIG_SIZE, AUTO_SAVE)
        
    print("\n全ての処理が完了しました。")