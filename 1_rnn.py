"""
Amaç: RNN kullanarak duygu analizi (sentiment analysis) yapmak.
    - sınıflandırma problemi: restoran yorumları, etiket: positive, negative
Adımlar:
    - gerkeli kütüphaneleri yükle (tensorflow, pytorch)
    - veri setini yükle (gemini ile veri üretme)
    - metin ön işleme (tokenization, paddingi label coding, train test split)
    - embedding katmanı oluştur word2vec
    - Rnn modeli oluştur (embedding -> simpleRNN -> Dense Layer)
    - modelin derlenmesi ve eğitimi
    - modelin değerlendirilmesi
    - user test (cümlelerin sınıflandırılması için fonksiyon yazma)

kurulum: pip install tensorflow numpy pandas scikit-learn gensim keras keras-preprocessing
"""

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer

from keras_preprocessing.sequence import pad_sequences

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# veri seti oluştur
# restoran yorumları ve etiketleri
reviews_data = {
    "text": [
        "Yemekler çok lezzetliydi, kesinlikle tekrar geleceğim!",
        "Servis hızı tam bir felaket, garson çağırmak için efor sarf ettik.",
        "Mezeler inanılmaz tazeydi, bayıldım.",
        "Fiyatlar lezzete göre aşırı abartılı, değmez.",
        "Garsonların ilgisi için teşekkür ederiz, çok naziktiler.",
        "Masanın örtüsü lekeliydi, hijyen konusunda sınıfta kaldılar.",
        "Ambiyans çok huzurlu, müzik seçimi şahane.",
        "Siparişimiz tam 1 saatte ancak masaya gelebildi.",
        "Hayatımda yediğim en iyi kebaptı, ellerinize sağlık.",
        "Porsiyonlar kuş kadar, doymadan masadan kalktık.",
        "Fiyatlar sundukları kaliteye göre çok makul.",
        "Müzik sesi o kadar yüksekti ki birbirimizi duyamadık.",
        "Tatlı ikramı bizi çok mutlu etti, çok ince bir düşünce.",
        "Rezervasyonumuz olmasına rağmen yarım saat kapıda bekletildik.",
        "Ailecek gidilecek en nezih yerlerden biri.",
        "Menüdeki çoğu yemek maalesef ellerinde kalmamıştı.",
        "Her şey kusursuzdu, 10 numara 5 yıldız bir mekan.",
        "Lavabolar çok kirli ve bakımsızdı, hiç yakışmamış.",
        "Hamburger ekmeği yumuşacıktı, soslar kendi yapımları ve harika.",
        "Gelen et çok sert ve kayış gibiydi, çiğnemek imkansızdı.",
        "Deniz manzarası eşliğinde yemek yemek büyük bir keyifti.",
        "Hesapta yanlışlık yaptılar, kontrol etmesek fazla ödeyecektik.",
        "Şefin özel tabağını mutlaka denemelisiniz, efsane!",
        "İçerisi çok basıktı, havalandırma sistemi düzgün çalışmıyor.",
        "Hijyen kurallarına çok dikkat ediyorlar, her yer tertemiz.",
        "Çatal ve bıçaklar kirli geldi, birkaç kez değiştirtmek zorunda kaldık.",
        "Hızlı servis ve sıcak yemekler; işte budur!",
        "Mezeler bozuk gibi kokuyordu, tadına bakmaya korktuk.",
        "Kahvaltısı çok çeşitli ve ürünler oldukça kaliteli.",
        "Vale hizmeti çok pahalı ve aracımızı çok geç getirdiler.",
        "Rezervasyon süreci çok kolaydı, bizi kapıda karşıladılar.",
        "Çocuklu aileler için hiç uygun değil, mama sandalyesi bile yok.",
        "Mekanın dekorasyonu çok şık ve modern tasarlanmış.",
        "Fotoğraflardaki yemeklerle masaya gelenlerin alakası yok.",
        "Etler lokum gibi pişmişti, ağızda dağılıyor.",
        "Çok gürültülü bir ortam, kafa dinlemek için asla gelmeyin.",
        "Vegan seçeneklerin olması benim için büyük bir artıydı.",
        "Tatlılar bayattı, muhtemelen birkaç günlük ürünlerdi.",
        "Arkadaş grubuyla eğlenmek için çok uygun bir atmosferi var.",
        "Siparişimizi yanlış getirdiler, üstüne bir de biz suçluymuşuz gibi davrandılar.",
        "Beklediğimize değdi, her kuruşunu hak eden bir işletme.",
        "Masalar birbirine çok yakın, özel alan diye bir şey kalmamış.",
        "Çorbaları ev yapımı tadında, çok lezzetli.",
        "Tam bir hayal kırıklığı, sosyal medya reklamlarına aldandık.",
        "İlgi ve alaka en üst seviyedeydi, kendimizi özel hissettik.",
        "Bir daha asla önünden bile geçmem, kimseye de önermem.",
        "Otopark sorunu olmaması büyük bir avantaj.",
        "Yemekler buz gibi geldi, mikrodalgada ısıtılmış gibiydi.",
        "Akşam yemeği için çok romantik ve loş bir ortamı var.",
        "Kesinlikle tavsiye ediyorum, pişman olmazsınız."
    ],
    "label": [
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "negative",
        "positive", "negative", "positive", "negative", "positive",
        "negative", "positive", "negative", "positive", "positive"
    ]
}

df = pd.DataFrame(reviews_data)
print(df.head())

# metin ön işleme
tokenizer = Tokenizer() # kelimeleri sayısal değerlere dönüştürmek için tokenizer kullanıyoruz
tokenizer.fit_on_texts(df["text"]) # kelime sözlüğünü oluşturuyoruz
text_sequences = tokenizer.texts_to_sequences(df["text"]) # yorumları sayısal dizilere dönüştürüyoruz
word_index = tokenizer.word_index # sözlük: kelime -> indeks
#print("Kelime Sözlüğü:", word_index)

# padding 
"""
max padding uzunluğu = 3 
"""
max_sequence_length = max(len(seq) for seq in text_sequences) # en uzun yorumun uzunluğunu buluyoruz
print("En Uzun Yorum Uzunluğu:", max_sequence_length)
X = pad_sequences(text_sequences, maxlen=max_sequence_length) # dizileri aynı uzunluğa getiriyoruz
print(f"giriş verisi (x) shape: {X.shape}")
print(X)

# label encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"]) # etiketleri sayısal değerlere dönüştürüyoruz positive -> 1, negative -> 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # veriyi eğitim ve test olarak bölüyoruz

# embedding katmanı oluştur
sentences= [text.split() for text in df["text"]] # yorumları kelime listelerine dönüştürüyoruz

#word2vec modelini eğitiyoruz
word2vec_model = Word2Vec(sentences, vector_size=50, window=5, min_count=1) 

embedding_dim = 50 # kelime vektörlerinin boyutu

#embedding matrisi oluşturuyoruz
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim)) 
for word, i in word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word] # kelime vektörlerini matrise yerleştiriyoruz

# RNN modeli oluştur
model = Sequential()

# embedding katmanı
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))

# RNN katmanı --> units: gizli katman sayısı, return_sequences: False ise sadece son çıktıyı döndürür, True ise tüm çıktıları döndürür
model.add(SimpleRNN(units=50, return_sequences=False))

# output katmanı
model.add(Dense(1, activation="sigmoid")) # binary classification için sigmoid aktivasyon fonksiyonu kullanıyoruz

# compile
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# modelin eğitilmesi
model.fit(X_train, y_train, epochs=10, # eğitim tekrar sayısı
           batch_size=2, # mini batch size
           validation_data=(X_test, y_test)) # test setini doğrulama için kullanıyoruz

# modelin değerlendirilmesi
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# yeni cümlelerin sınıflandırılması için fonksiyon yazma
def classify_sentence(sentence):
    """
    Verilen bir cümlenin duygu analizini yapar ve sınıflandırır.
    """
    seq = tokenizer.texts_to_sequences([sentence]) # cümleyi sayısal diziye dönüştür
    padded_seq = pad_sequences(seq, maxlen=max_sequence_length) # diziyi padding yap
    prediction = model.predict(padded_seq) # modeli kullanarak tahmin yap
    prediced_class= (prediction > 0.5).astype(int) # tahmin sonucunu sınıflandır

    label= "positive" if prediced_class[0][0] == 1 else "negative"
    return label

new_sentence = "Yemekler çok güzeldi, servis mükemmeldi!"
result =classify_sentence(new_sentence)
print(f"Cümle: '{new_sentence}' --> Sınıflandırma: {result}")