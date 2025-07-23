import h5py

hdf5_file = '/home/learning-larr/bc_data/gello/0718_201746/data.hdf5'

with h5py.File(hdf5_file, 'r') as f:
    data_group = f['data']

    # 1) 데이터셋(key) 목록 확인
    keys = list(data_group.keys())
    print("=== Available datasets ===")
    for k in keys:
        print(f"  • {k}")
    print()

    # 2) 각 데이터셋의 shape, dtype 출력
    print("=== Dataset shapes and dtypes ===")
    for k in keys:
        ds = data_group[k]
        print(f"{k:20s} → shape: {ds.shape}, dtype: {ds.dtype}")
    print()

    # 3) (원한다면) 실제 내용을 일부 출력
    print("=== Sample data preview ===")
    for k in keys:
        arr = data_group[k][:]
        # 배열이 너무 크면 슬라이스해서 보여 줍니다
        preview = arr.flat[:5]  # 앞의 5개 원소
        print(f"{k:20s} → first 5 elements: {preview}")
