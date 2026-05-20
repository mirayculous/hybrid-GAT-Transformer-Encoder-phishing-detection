# Hasil Pelatihan Model GATv2 (Skenario Split 90:10)

Proses pelatihan dilakukan menggunakan GPU (CUDA) selama 50 *epoch*. Berikut adalah rekaman log dari proses *training* dan *validation*:

<details>
<summary><b>Klik untuk melihat Log Training selengkapnya (Epoch 1 - 50)</b></summary>

```text
Menggunakan device: cuda

Mulai Training...

Epoch: 001 | Train Loss: 0.2940 | Val Loss: 0.2124 | Val Acc: 92.19% | Val AUC: 0.9694
Epoch: 002 | Train Loss: 0.2486 | Val Loss: 0.2018 | Val Acc: 91.78% | Val AUC: 0.9740
Epoch: 003 | Train Loss: 0.2412 | Val Loss: 0.2193 | Val Acc: 91.63% | Val AUC: 0.9745
Epoch: 004 | Train Loss: 0.2355 | Val Loss: 0.1979 | Val Acc: 92.09% | Val AUC: 0.9748
Epoch: 005 | Train Loss: 0.2366 | Val Loss: 0.2100 | Val Acc: 92.19% | Val AUC: 0.9742
Epoch: 006 | Train Loss: 0.2357 | Val Loss: 0.1949 | Val Acc: 92.24% | Val AUC: 0.9759
Epoch: 007 | Train Loss: 0.2322 | Val Loss: 0.1942 | Val Acc: 92.54% | Val AUC: 0.9757
Epoch: 008 | Train Loss: 0.2324 | Val Loss: 0.2323 | Val Acc: 90.22% | Val AUC: 0.9745
Epoch: 009 | Train Loss: 0.2297 | Val Loss: 0.1960 | Val Acc: 92.39% | Val AUC: 0.9763
Epoch: 010 | Train Loss: 0.2294 | Val Loss: 0.1915 | Val Acc: 92.64% | Val AUC: 0.9768
Epoch: 011 | Train Loss: 0.2287 | Val Loss: 0.2175 | Val Acc: 92.04% | Val AUC: 0.9765
Epoch: 012 | Train Loss: 0.2262 | Val Loss: 0.2275 | Val Acc: 91.53% | Val AUC: 0.9760
Epoch: 013 | Train Loss: 0.2267 | Val Loss: 0.1940 | Val Acc: 91.99% | Val AUC: 0.9762
Epoch: 014 | Train Loss: 0.2249 | Val Loss: 0.1944 | Val Acc: 92.34% | Val AUC: 0.9761
Epoch: 015 | Train Loss: 0.2242 | Val Loss: 0.1853 | Val Acc: 92.69% | Val AUC: 0.9781
Epoch: 016 | Train Loss: 0.2225 | Val Loss: 0.1949 | Val Acc: 91.94% | Val AUC: 0.9753
Epoch: 017 | Train Loss: 0.2233 | Val Loss: 0.1990 | Val Acc: 92.69% | Val AUC: 0.9776
Epoch: 018 | Train Loss: 0.2207 | Val Loss: 0.1896 | Val Acc: 92.39% | Val AUC: 0.9778
Epoch: 019 | Train Loss: 0.2215 | Val Loss: 0.2102 | Val Acc: 89.92% | Val AUC: 0.9766
Epoch: 020 | Train Loss: 0.2208 | Val Loss: 0.2044 | Val Acc: 92.44% | Val AUC: 0.9738
Epoch: 021 | Train Loss: 0.2195 | Val Loss: 0.2197 | Val Acc: 91.48% | Val AUC: 0.9771
Epoch: 022 | Train Loss: 0.2189 | Val Loss: 0.1955 | Val Acc: 92.44% | Val AUC: 0.9763
Epoch: 023 | Train Loss: 0.2227 | Val Loss: 0.1925 | Val Acc: 92.69% | Val AUC: 0.9783
Epoch: 024 | Train Loss: 0.2207 | Val Loss: 0.1867 | Val Acc: 92.84% | Val AUC: 0.9785
Epoch: 025 | Train Loss: 0.2206 | Val Loss: 0.1837 | Val Acc: 92.64% | Val AUC: 0.9786
Epoch: 026 | Train Loss: 0.2199 | Val Loss: 0.1932 | Val Acc: 92.94% | Val AUC: 0.9775
Epoch: 027 | Train Loss: 0.2201 | Val Loss: 0.1981 | Val Acc: 90.98% | Val AUC: 0.9785
Epoch: 028 | Train Loss: 0.2206 | Val Loss: 0.1832 | Val Acc: 93.20% | Val AUC: 0.9787
Epoch: 029 | Train Loss: 0.2176 | Val Loss: 0.1817 | Val Acc: 92.39% | Val AUC: 0.9785
Epoch: 030 | Train Loss: 0.2176 | Val Loss: 0.1810 | Val Acc: 92.74% | Val AUC: 0.9791
Epoch: 031 | Train Loss: 0.2184 | Val Loss: 0.2005 | Val Acc: 91.33% | Val AUC: 0.9779
Epoch: 032 | Train Loss: 0.2210 | Val Loss: 0.2017 | Val Acc: 91.94% | Val AUC: 0.9786
Epoch: 033 | Train Loss: 0.2180 | Val Loss: 0.1878 | Val Acc: 92.74% | Val AUC: 0.9780
Epoch: 034 | Train Loss: 0.2194 | Val Loss: 0.1860 | Val Acc: 92.49% | Val AUC: 0.9784
Epoch: 035 | Train Loss: 0.2179 | Val Loss: 0.2033 | Val Acc: 92.34% | Val AUC: 0.9790
Epoch: 036 | Train Loss: 0.2199 | Val Loss: 0.1880 | Val Acc: 92.89% | Val AUC: 0.9788
Epoch: 037 | Train Loss: 0.2179 | Val Loss: 0.2061 | Val Acc: 91.99% | Val AUC: 0.9786
Epoch: 038 | Train Loss: 0.2145 | Val Loss: 0.1810 | Val Acc: 92.59% | Val AUC: 0.9791
Epoch: 039 | Train Loss: 0.2201 | Val Loss: 0.1903 | Val Acc: 92.64% | Val AUC: 0.9782
Epoch: 040 | Train Loss: 0.2199 | Val Loss: 0.1953 | Val Acc: 92.49% | Val AUC: 0.9784
Epoch: 041 | Train Loss: 0.2150 | Val Loss: 0.2167 | Val Acc: 91.73% | Val AUC: 0.9775
Epoch: 042 | Train Loss: 0.2168 | Val Loss: 0.1851 | Val Acc: 92.94% | Val AUC: 0.9790
Epoch: 043 | Train Loss: 0.2195 | Val Loss: 0.1860 | Val Acc: 92.84% | Val AUC: 0.9782
Epoch: 044 | Train Loss: 0.2157 | Val Loss: 0.1845 | Val Acc: 92.54% | Val AUC: 0.9793
Epoch: 045 | Train Loss: 0.2189 | Val Loss: 0.2029 | Val Acc: 91.68% | Val AUC: 0.9764
Epoch: 046 | Train Loss: 0.2153 | Val Loss: 0.1817 | Val Acc: 92.89% | Val AUC: 0.9786
Epoch: 047 | Train Loss: 0.2190 | Val Loss: 0.2027 | Val Acc: 91.89% | Val AUC: 0.9783
Epoch: 048 | Train Loss: 0.2148 | Val Loss: 0.1847 | Val Acc: 92.49% | Val AUC: 0.9780
Epoch: 049 | Train Loss: 0.2162 | Val Loss: 0.1960 | Val Acc: 92.24% | Val AUC: 0.9782
Epoch: 050 | Train Loss: 0.2154 | Val Loss: 0.1925 | Val Acc: 92.34% | Val AUC: 0.9781
```
</details>


## === HASIL EVALUASI DATA TEST ===
Accuracy : 91.38%<br/>
Precision: 95.42%<br/>
Recall   : 84.75%<br/>
F1-Score : 89.77%<br/>
AUC-ROC  : 0.9676
```

![](results/kurva training validation loss 90_10.png)
![](results/confusion matrix ROC 90_10.png)

