# app/CUDA_kontrol.py

import torch

def cuda_durumu():
    print("=== CUDA DURUMU ===")
    print(f"CUDA kullanılabilir mi? : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Kullanılabilir CUDA aygıtı sayısı : {torch.cuda.device_count()}")
        print(f"Aktif CUDA aygıtı            : {torch.cuda.current_device()}")
        print(f"Aktif Aygıt İsmi             : {torch.cuda.get_device_name(torch.cuda.current_device())}")
        print(f"Toplam Bellek (MB)           : {round(torch.cuda.get_device_properties(0).total_memory / (1024**2))}")
    else:
        print("CUDA desteklenmiyor veya sürücü düzgün yüklenmemiş.")

if __name__ == "__main__":
    cuda_durumu()