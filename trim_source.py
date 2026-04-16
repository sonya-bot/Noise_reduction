import os
import librosa
import soundfile as sf # 音声ファイルの保存に使用します

def process_trimming(input_dir, output_dir, start_sec=5.0, duration=30.0):
    """
    指定フォルダ内のWAVファイルを一括でトリミングして保存する関数
    
    start_sec: 録音開始から何秒間を捨てるか（デフォルト5秒）
    duration: 切り出す長さ（デフォルト30秒）
    """
    # 出力先フォルダが存在しない場合は自動作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 対象ファイルのリストアップ
    audio_extensions = ['.wav']
    audio_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in audio_extensions]
    
    if not audio_files:
        print(f"エラー: フォルダ '{input_dir}' に音声ファイルが見つかりません。")
        return

    print(f"音声データのトリミング加工を開始します")
    print(f"設定: 開始 {start_sec} 秒地点から {duration} 秒間を切り出し\n")

    for file_name in audio_files:
        input_path = os.path.join(input_dir, file_name)
        
        base_name = os.path.splitext(file_name)[0]
        output_name = f"{base_name}.wav"
        output_path = os.path.join(output_dir, output_name)
        
        print(f"処理中: {file_name}")
        
        try:
            # 音声の読み込み (元のサンプリングレートを維持)
            y, sr = librosa.load(input_path, sr=None)
            
            # 秒数をサンプル数（配列のインデックス）に変換
            start_sample = int(start_sec * sr)
            end_sample = int((start_sec + duration) * sr)
            
            # トリミング実行
            # ※もし元の音声が「開始秒数＋30秒」より短い場合は、処理を行わずスキップ
            if len(y) < end_sample:
                print(f"警告: 元データが{duration}秒未満のため、スキップします")
                pass
            else:
                y_trimmed = y[start_sample:end_sample]
                
            # 加工した音声をWAVファイルとして書き出し
            sf.write(output_path, y_trimmed, sr)
            print(f"保存完了: {output_name} (長さ: {len(y_trimmed)/sr:.2f} 秒)")
            
        except Exception as e:
            print(f"エラー: {file_name} の処理中に問題が発生しました: {e}")

    print("\n全ての処理が完了しました")

if __name__ == "__main__":
    # --- パラメータ設定 ---
    INPUT_PATH = "/Users/Souma/Develop/exp_data/recording_data/raw_data/scene_4"             # 元のデータが入っているフォルダ
    OUTPUT_PATH = "/Users/Souma/Develop/exp_data/recording_data/processed_data/scene_4"  # 加工後の30秒データを保存するフォルダ
    
    # 自転車に乗り始めるまでの「準備時間」に合わせてここを調整してください
    START_SECONDS = 5.0  # 最初から5秒間を捨てる
    DURATION_SECONDS = 30.0 # 切り出し区間(秒)
    
    process_trimming(INPUT_PATH, OUTPUT_PATH, START_SECONDS, DURATION_SECONDS)