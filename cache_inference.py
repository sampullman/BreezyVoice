from single_inference import main_cached

if __name__ == "__main__":
    '''
    test command:

    python3 cache_inference.py --spk_id 臺灣女 \
        --content_to_synthesize "歡迎使用聯發創新基地 BreezyVoice 模型。" \
        --output_path results/output.wav
    '''
    main_cached()
