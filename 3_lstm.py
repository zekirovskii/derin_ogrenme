"""
Amaç: LSTM tabanlı bşrmidl modeli ile metin üretimi yapmak.
Eğitim verisi olarak Gemini ile oluşturulmuuş günlük ifadeler kullanılmıştır.
model verilen bir başlangıç kelimesinden yeni kelimeler ya da cümleler üretir.

Adımlar:
    - eğitim verisi oluşturma
    - tokenizasyon: kelimeleri sayısal vektörlere dönüştürme
    - n-gram dizileri oluşturma: dil modeli için girdi ve çıktı çiftleri hazırlama
    - padding --> tüm dizileri aynı uzunluğa getirme
    - LSTM tabanlı model oluşturma
    - modelin derlenmesi ve eğitilmesi
    - yeni metin üretimi için fonksiyon yazılması

    kurulum: pip install numpy tensorflow keras 
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#eğitim verisi oluşturma
texts = [
    "Bugün hava çok güzel, dışarıda yürüyüş yapmayı düşünüyorum.",
    "Kitap okumak beni gerçekten mutlu ediyor.",
    "Sabah kalkınca ilk işim güzel bir kahve demlemek oluyor.",
    "Bugün yapılacaklar listem bir hayli kabarık.",
    "Akşam yemeğinde ne pişireceğime henüz karar veremedim.",
    "En sevdiğim dizinin yeni bölümü bu akşam yayınlanacak.",
    "Hafta sonu arkadaşlarla sahilde buluşmak için sözleştik.",
    "Yarınki toplantı için hazırladığım sunumu tekrar gözden geçirmeliyim.",
    "Evin içindeki bitkileri sulama zamanı geldi de geçiyor.",
    "Bugün metro beklediğimden çok daha kalabalıktı.",
    "Market alışverişi yaparken listeye sadık kalmaya çalışıyorum.",
    "Yeni aldığım kulaklığın ses kalitesi gerçekten harika.",
    "Spor salonuna gitmek için kendimde o enerjiyi bulmaya çalışıyorum.",
    "Akşam serinliğinde balkonda çay içmenin keyfi bir başka.",
    "Telefonumun şarjı bitmek üzere, hemen bir priz bulmalıyım.",
    "Bugün ofiste işler beklediğimden çok daha hızlı ilerledi.",
    "Hafta sonu için küçük bir doğa yürüyüşü planlıyoruz.",
    "En son okuduğum kitabın sonu beni çok şaşırttı.",
    "Mutfağı toparladıktan sonra kendime bir dinlenme molası vereceğim.",
    "Yolda yürürken eski bir arkadaşıma rastladım ve ayaküstü sohbet ettik.",
    "Kışlık kıyafetleri dolaptan çıkarma vaktim geldi.",
    "Bugün kendimi biraz yorgun hissediyorum, erken uyumayı planlıyorum.",
    "İnternetten verdiğim siparişin bugün gelmesini bekliyorum.",
    "Sabah erken kalkınca günün çok daha verimli geçtiğini fark ettim.",
    "Güneş gözlüğümü evde unuttuğum için tüm gün gözlerimi kısmak zorunda kaldım.",
    "Yeni bir yemek tarifi denedim ve sonuç beklediğimden çok daha iyi oldu.",
    "Arabayı yıkatmam gerekiyor ama hava her an bozabilir gibi duruyor.",
    "Bugün işe giderken en sevdiğim podcast'in yeni bölümünü dinledim.",
    "Çalışma masamı düzenlemek odaklanmamı gerçekten kolaylaştırıyor.",
    "Günün stresini atmak için akşam uzun bir duş aldım.",
    "Pazar kahvaltısı için taze ekmek almak üzere fırına gittim.",
    "Dolabı temizlerken tarihinin geçtiğini fark ettiğim birçok şey çıktı.",
    "Bilgisayarımın güncelleme yapması tam da en işim olduğu ana denk geldi.",
    "Bugün gökyüzündeki bulutların şekilleri gerçekten çok etkileyiciydi.",
    "Biraz kafa dinlemek için telefonumu bir süreliğine uçak moduna aldım.",
    "Kütüphaneye gidip sessiz bir ortamda çalışmak bana iyi geliyor.",
    "Evdeki eski fotoğraflara bakarken zamanın ne kadar hızlı geçtiğini anladım.",
    "Bugün trafikte çok fazla vakit kaybettim, keşke toplu taşıma kullansaydım.",
    "Bahçedeki çiçekler baharın gelmesiyle birlikte harika görünmeye başladı.",
    "Diş randevuma geç kalmamak için evden biraz erken çıktım.",
    "Akşam üzerine doğru acıkınca kendime küçük bir atıştırmalık hazırladım.",
    "Yeni başladığım hobi sayesinde kendimi çok daha üretken hissediyorum.",
    "Ev arkadaşımla oturup saatlerce gelecek planlarımız hakkında konuştuk.",
    "Cüzdanımı çantamın içinde bulamayınca kısa süreli bir panik yaşadım.",
    "Bugün spor yaparken en sevdiğim hareketli şarkıları dinledim.",
    "Tatil planları yapmak insanı iş stresinden biraz olsun uzaklaştırıyor.",
    "Kedim bütün gün koltuğun üzerinde mışıl mışıl uyudu.",
    "Bugün öğle yemeğinde dışarıdan söylemek yerine evde sandviç hazırladım.",
    "Odama yeni bir tablo asınca odanın havası bir anda değişti.",
    "Yarınki sınav için son tekrarlarımı yapıp erkenden yatacağım.",
    "Üzerime kahve dökülünce tişörtümü değiştirmek için eve dönmek zorunda kaldım.",
    "Eczaneden vitamin almak için yolumu biraz uzattım.",
    "Bugün aldığım bir haber beni gerçekten çok heyecanlandırdı.",
    "Bisiklet sürmek hem spor oluyor hem de şehri keşfetmemi sağlıyor.",
    "Günün sonunda yatağa uzanıp tavana bakarak hayal kurmayı seviyorum.",
    "Haftalık temizliği bitirince evin mis gibi kokması çok huzur verici.",
    "Kumandanın pilleri bitmiş, manuel olarak kanalları değiştirmek zorunda kaldım.",
    "Yağmurun cama vuran sesi eşliğinde uyumak paha biçilemez.",
    "Bugün iş görüşmem vardı ve beklediğimden çok daha pozitif geçti.",
    "Kendi ekmeğimi evde yapmaya başladığımdan beri dışarıdan almıyorum.",
    "Doğum günü pastası için gerekli malzemeleri listeye ekledim.",
    "Bugün metroda kitap okuyan çok fazla insan görmek beni mutlu etti.",
    "Bilgisayar başında çok fazla oturduğum için sırtım ağrımaya başladı.",
    "Gitarımın tellerini değiştirdikten sonra sesi çok daha temiz çıkıyor.",
    "Akşam yemeğinden sonra kısa bir mahalle yürüyüşüne çıktım.",
    "Bugün kendime bir hediye alıp uzun zamandır istediğim o kazağı aldım.",
    "Yaz tatili için valizimi şimdiden hazırlamaya başlamak istiyorum.",
    "Kuşların sabah cıvıltısıyla uyanmak alarm sesinden çok daha güzel.",
    "Bugün şans eseri çok uygun fiyata çok güzel bir ayakkabı buldum.",
    "Mutfak robotu bozulunca işlerimi elde halletmek zorunda kaldım.",
    "Eski bir diziyi tekrar izlemek bana kendimi çocukluğumda gibi hissettiriyor.",
    "Bugün çok yoğun geçtiği için kendime vakit ayırmayı tamamen unuttum.",
    "Rüyamda gördüğüm şeyin etkisinden tüm gün çıkamadım.",
    "Yeni bir dil öğrenmeye başlamak başta zor olsa da çok keyifli.",
    "Yemekten sonra bulaşıkları hemen yıkamak mutfağın düzenli kalmasını sağlıyor.",
    "Bugün hava durumu yağmurlu gösteriyordu ama tüm gün güneş açtı.",
    "Saksıdaki toprağı değiştirirken balkon biraz kirlendi ama değdi.",
    "Gece geç saatte gelen o acıkma hissiyle buzdolabının önünde buluştuk.",
    "Spor ayakkabılarımı makinede yıkayınca ilk günkü gibi bembeyaz oldular.",
    "Bugün otobüste yanıma oturan teyze bana çok tatlı bir hikaye anlattı.",
    "Gereksiz e-postaları silmek dijital dünyamı biraz daha sadeleştirdi.",
    "Faturayı ödemeyi unutunca internetim kısa süreliğine kesildi.",
    "Akşam yemeğine misafir geleceği için evi biraz daha özenli topladım.",
    "Yeni bir parfüm denedim ve kokusu tüm gün üzerimde kaldı.",
    "Bugün kendimi geliştirmek adına yeni bir makale okudum.",
    "Yolda gördüğüm yavru kediyi sevmek günün tüm yorgunluğunu aldı.",
    "Sabah duş almak güne daha zinde başlamama yardımcı oluyor.",
    "Bugün işten biraz erken çıkıp kendime vakit ayırmaya karar verdim.",
    "Dolaptaki meyveler bozulmadan bir meyve salatası yapmalıyım.",
    "Yatağımı toplamak odanın genel havasını bir anda toplu gösteriyor.",
    "Eski bir radyo buldum ve kanalları ararken çocukluğuma döndüm.",
    "Bugün çok konuşmak zorunda kaldığım için sesim biraz kısıldı.",
    "Hava kararınca şehrin ışıklarını izlemek beni çok sakinleştiriyor.",
    "Yeni bir teknolojik alet alınca kılavuzunu okumadan kurmaya çalışıyorum.",
    "Akşam çayının yanına en sevdiğim bisküvileri aldım.",
    "Bugün marketteki indirimleri takip ederek alışveriş yaptım.",
    "Ayakkabımın bağcığı kopunca geçici bir çözüm üretmek zorunda kaldım.",
    "En sevdiğim şarkı radyoda çıkınca sesini sonuna kadar açtım.",
    "Bugün sadece kendim için bir şeyler yapmak bana çok iyi geldi.",
    "Yarın yepyeni bir gün olacak ve ben buna hazırım."
]

# preprocessing
#tokenizasyon: kelimeleri sayısal vektörlere dönüştürme
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts) 
total_words = len(tokenizer.word_index) + 1 # kelime sayısı (index 0 kullanılmaz)

#ngram dizileri oluşturma: dil modeli için girdi ve çıktı çiftleri hazırlama
# örn: "Bugün hava çok güzel" --> ["Bugün", "hava"], ["Bugün", "hava", "çok"], ["Bugün", "hava", "çok", "güzel"]
input_sequences = []
for text in texts:
    token_list = tokenizer.texts_to_sequences([text])[0] # metni sayısal diziye dönüştür
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1] # n-gram dizisi oluştur
        input_sequences.append(n_gram_sequence)

# en uzun n-gram dizisinin uzunluğunu bulma
max_sequence_length = max(len(seq) for seq in input_sequences)

# padding --> tüm dizileri aynı uzunluğa getirme
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding="pre")

# girdi (x) ve hedef (y)
X = input_sequences[:,:-1] # girdi dizileri (son kelime hariç)
y = input_sequences[:,-1] # hedef kelimeler (son kelime)

#one-hot encoding
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# model oluşturma
model = Sequential()

# embedding katmanı
# total_words: kelime sayısı, embedding_dim: kelime vektörlerinin boyutu, input_length: girdi dizilerinin uzunluğu
model.add(Embedding(total_words, 50, input_length=X.shape[1]))

model.add(LSTM(100, return_sequences=False)) # LSTM katmanı, units: gizli katman sayısı

# çıkış katmanı
model.add(Dense(total_words, activation="softmax")) # çok sınıflı sınıflandırma için softmax aktivasyon fonksiyonu kullanıyoruz

# modelin derlenmesi
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# modelin eğitilmesi
model.fit(X, y, epochs=100, verbose=1)

# metnin üretimi

def generate_text(seed_text, next_words):
    """
    seed_text: başlangıç kelimesi veya cümlesi
    next_words: üretilecek kelime sayısı
    """

    for _ in range(next_words):
        # girdi metinini sayısal diziye dönüştür
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        # padding
        token_list = pad_sequences([token_list], maxlen=max_sequence_length - 1, padding="pre")
        # modelden olasılık dağılımını alalım

        predicted = model.predict(token_list, verbose=0)

        # en yüksek olasılığa sahip kelimenin indexini bulalım
        predicted_index = np.argmax(predicted, axis=-1)

        # indexi kelimeye dönüştürelim
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break   

        # üretilen kelimeyi seed_text'e ekleyelim
        seed_text += " " + output_word
    return seed_text

seed_text = "Bugün hava"
print(generate_text(seed_text, next_words=5))