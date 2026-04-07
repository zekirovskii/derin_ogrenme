"""
Amaç: IMDB film yorumları veri seti kullanarak GRU tabanlı bir modelle duygu analizi (sentiment classification) yapmak.

Adımlar:
    - gerekli kütüphaneler
    - IMDB veri seti yükle
    - padding
    - GRU tabanlı model kurulması
    - Modelin derlenmesi ve eğitilmesi
    - Modelin test edilmesi (evulation)
    - yeni yorum tahmini için fonksiyon yazılması

Kurulum:
pip install tensorflow numpy pandas keras
"""

#import libraries
import numpy as np
from tensorflow.keras.datasets import imdb #IMDB veri setini yüklemek için kullanılır
from tensorflow.keras.preprocessing.sequence import pad_sequences #veri setindeki yorumların uzunluklarını eşitlemek için kullanılır

from tensorflow.keras.models import Sequential #model oluşturmak için kullanılır
from tensorflow.keras.layers import Embedding, GRU, Dense #model katmanları için kullanılır

# veri setini yükle
# imdb veri setinde en sık geçen 10000 kelime

num_words = 10000
max_sequence_length = 200

# X_train ve X_test = yorumlar
# y_train ve y_test = etiketler (0: negatif, 1: pozitif)
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)
print(f"Train boyutu: {len(X_train)}, Test boyutu: {len(X_test)}")

# padding
X_train_padded = pad_sequences(X_train, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test, maxlen=max_sequence_length) # yorumların uzunluklarını eşitliyoruz

print(f"X_train şekli: {X_train_padded.shape}, X_test şekli: {X_test_padded.shape}")

"""
Train boyutu: 25000, Test boyutu: 25000
X_train şekli: (25000, 200), X_test şekli: (25000, 200)
"""
# GRU tabanlı model oluştur
embedding_dim = 100 # kelime vektörlerinin boyutu

model = Sequential()

# embedding katmanı
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=max_sequence_length)) 

# GRU katmanı
# gizli katman sayısı, return_sequences: False ise sadece son çıktıyı döndürür, True ise tüm çıktıları döndürür
model.add(GRU(units=64,return_sequences=False)) # GRU katmanı

# output katmanı
#dense (1) ikili sınıflandırma için kullanılır, sigmoid aktivasyon fonksiyonu ile 0-1 arasında bir çıktı verir
model.add(Dense(1, activation="sigmoid")) 

# modelin derlenmesi
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
print(model.summary())

# modelin eğitilmesi
model.fit(X_train_padded, y_train, epochs=3, batch_size=128, validation_split=0.2, verbose=1)

# modelin değerlendirilmesi
test_loss, test_accuracy = model.evaluate(X_test_padded, y_test , verbose=1)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# yeni yorum tahmini için fonksiyon yazılması
# imdb dataset sayısal indexe dönüştürülmüş yorumlar içerir, bu yüzden yeni yorumları sayısal indexe dönüştürmek için word_index kullanılır. index_to_word mapping yapılmalı

word_index = imdb.get_word_index() # kelime -> index mapping
index_to_word = {index + 3: word for word, index in word_index.items()} # index -> kelime mapping (index 0, 1, 2 özel tokenlar için ayrılmıştır)

index_to_word[0] = "<PAD>" # padding tokenı
index_to_word[1] = "<START>" # imdb veri setinde her yorumun başında <START> tokenı bulunur, bu yüzden index 1'e atanır
index_to_word[2] = "<UNK>" # imdb veri setinde nadir kelimeler <UNK> tokenı ile temsil edilir, bu yüzden index 2'ye atanır

def decode_review(encoded_review):
    """
    sayısal bir yorumu tekrar metne dönüştürür.
    """
    return ' '.join([index_to_word.get(index, "<UNK>") for index in encoded_review])

def classify_review(review):
    """
    Verilen bir yorumun duygu analizini yapar ve sınıflandırır.
    """
    padded = pad_sequences([review], maxlen=max_sequence_length) # yorumu sayısal indexe dönüştürüp padding yapar
    prob= model.predict(padded)[0][0] # modelin tahmin ettiği olasılığı alır
    label = "positive" if prob > 0.5 else "negative" # olasılığa göre sınıflandırma yapar
    return label, prob

# örnek bir yorumun sınıflandırılması
decoded=decode_review(X_test[0]) # sayısal indexe dönüştürülmüş bir yorumu tekrar metne dönüştür
print(f"Decoded review: {decoded}")
predicted_label, predicted_prob = classify_review(X_test[0]) # yorumu sınıflandır
print(f"Tahmin: {predicted_label}, Olasılık: {predicted_prob:.4f}")