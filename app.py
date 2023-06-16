from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import os
import pickle
from io import BytesIO
import keras.utils as image
from PIL import Image

app = Flask(__name__)

model = tf.keras.models.load_model('NewData_Fixed_MobileNetV2_1e-4_80Split_Adam_Dense_with02Dropout.h5')

class_names = ["hdpe_container", "hdpe_botol", "ps", "pp_tutup_botol", "pp_botol", "pp_sedotan", "pet", "pp_container", "ldpe_bag", "ldpe_botol", "hdpe_tutup_botol"]

plastic_dict = {
    'pet': {
        'kode': 'pet',
        'nama': 'Polyethylene Terephthalate',
        'namalain': 'Polyethylene Terephthalate',
        'manfaat': 'manfaat dari PET (Polyethylene Terephthalate) sebagai bahan kemasan termasuk kekuatannya yang ringan dan tahan lama, transparansinya yang memungkinkan konsumen melihat produk di dalamnya, serta kemampuan untuk didaur ulang dan mengurangi limbah plastik.',
        'kekurangan': 'PET juga memiliki beberapa kekurangan. Salah satunya adalah potensi pelepasan zat berbahaya jika terkena suhu tinggi atau zat yang tidak cocok. Selain itu, PET rentan terhadap goresan dan deformasi, dan penggunaan bahan bakar fosil dalam produksi PET berkontribusi terhadap emisi gas rumah kaca. Terakhir, terdapat kemungkinan migrasi zat dari produk ke dalam makanan atau minuman yang dikemas dalam botol PET.',
        'logo': 'https://storage.googleapis.com/trashurebucket/dic_ml/Symbol_Resin_Code_1_PETE.svg'
    },
    'hdpe': {
        'kode': 'hdpe',
        'nama': 'High density polyethylene',
        'namalain': 'High-Density Polyethylene',
        'manfaat': 'manfaat HDPE (High-Density Polyethylene) sebagai bahan plastik termasuk ketahanan terhadap bahan kimia, seperti asam dan alkali, serta tahan terhadap kelembaban dan air. HDPE juga dapat didaur ulang untuk mengurangi limbah plastik dan memiliki kekuatan tarik yang tinggi, membuatnya cocok untuk aplikasi yang membutuhkan kekuatan dan keandalan.',
        'kekurangan': 'HDPE juga memiliki beberapa kekurangan. Misalnya, HDPE rentan terhadap suhu tinggi dan dapat deformasi atau meleleh jika terpapar suhu yang ekstrem. Selain itu, HDPE tidak transparan, sehingga tidak cocok untuk produk yang memerlukan transparansi atau tampilan visual. HDPE juga tidak tahan terhadap panas dan dapat retak jika terkena tekanan atau gaya bengkok yang ekstrem.',
        'logo': 'https://storage.googleapis.com/trashurebucket/dic_ml/Symbol_Resin_Code_2_HDPE.svg'
    },
    'pvc': {
        'kode': 'pvc',
        'nama': 'Polyvinyl chloride',
        'namalain': 'Polyvinyl chloride',
        'manfaat': 'PVC (Polyvinyl Chloride) memiliki beberapa manfaat yang membuatnya populer dalam berbagai aplikasi. Pertama, PVC memiliki ketahanan terhadap zat kimia, yang membuatnya cocok untuk digunakan dalam industri kimia dan kemasan makanan. Kedua, PVC tahan terhadap kelembaban dan tidak terpengaruh oleh air, sehingga sering digunakan dalam aplikasi pipa dan saluran air. Ketiga, PVC relatif tahan terhadap api dan memiliki sifat isolasi listrik yang baik, sehingga banyak digunakan dalam instalasi kabel dan produk elektronik.',
        'kekurangan': 'PVC juga memiliki beberapa kekurangan. Pertama, dalam produksinya, PVC menggunakan bahan dasar klorin yang dapat menghasilkan zat berbahaya, seperti dioksina, yang merupakan zat polutan berbahaya. Kedua, PVC sulit didaur ulang dan memiliki masa hancur yang lama dalam lingkungan. Hal ini berkontribusi terhadap masalah limbah plastik. Ketiga, jika terbakar, PVC dapat menghasilkan gas beracun dan zat kimia yang berbahaya bagi kesehatan manusia.',
        'logo': 'https://storage.googleapis.com/trashurebucket/dic_ml/Symbol_Resin_Code_3_V.svg'
    },
    'ldpe': {
        'kode': 'ldpe',
        'nama': 'Low density Polyethylene',
        'namalain': 'Low-Density Polyethylene',
        'manfaat': 'LDPE (Low-Density Polyethylene) memiliki beberapa manfaat yang membuatnya banyak digunakan dalam industri kemasan dan aplikasi lainnya. Pertama, LDPE memiliki sifat yang fleksibel dan elastis, sehingga mudah dibentuk dan cocok untuk berbagai produk, seperti kantong plastik dan wadah fleksibel. Kedua, LDPE memiliki ketahanan yang baik terhadap bahan kimia, sehingga sering digunakan dalam kemasan makanan dan produk kimia. Ketiga, LDPE memiliki sifat isolasi yang baik terhadap panas dan listrik, membuatnya cocok untuk penggunaan dalam pembungkus termal dan kabel listrik.',
        'kekurangan': 'LDPE juga memiliki beberapa kekurangan yang perlu dipertimbangkan. Pertama, LDPE memiliki ketahanan yang rendah terhadap suhu tinggi, sehingga dapat meleleh atau terdeformasi jika terpapar panas secara ekstrem. Kedua, LDPE tidak memiliki kekuatan tarik yang tinggi dibandingkan dengan beberapa jenis plastik lainnya. Ini dapat menjadi pertimbangan penting dalam aplikasi yang memerlukan kekuatan yang tinggi atau ketahanan terhadap tekanan mekanis. Ketiga, LDPE memiliki masa hancur yang lama dalam lingkungan dan sulit didaur ulang dengan efisiensi tinggi.',
        'logo': 'https://storage.googleapis.com/trashurebucket/dic_ml/Symbol_Resin_Code_4_LDPE.svg'
    },
    'pp': {
        'kode': 'pp',
        'nama': 'Polypropylene',
        'namalain': 'Polypropylene',
        'manfaat': 'PP (Polypropylene) memiliki beberapa manfaat yang membuatnya populer dalam berbagai aplikasi. Pertama, PP memiliki kekuatan yang tinggi dan tahan terhadap tekanan dan benturan, menjadikannya pilihan yang baik untuk produk-produk yang membutuhkan kekuatan mekanis yang tinggi. Kedua, PP memiliki sifat tahan terhadap bahan kimia, seperti asam dan alkali, sehingga sering digunakan dalam industri kimia. Ketiga, PP memiliki sifat tahan terhadap panas dan memiliki titik leleh yang tinggi, sehingga cocok untuk penggunaan dalam produk-produk yang terekspos suhu tinggi.',
        'kekurangan': 'PP juga memiliki beberapa kekurangan yang perlu dipertimbangkan. Pertama, PP rentan terhadap paparan sinar UV dan dapat mengalami degradasi jika terpapar langsung sinar matahari untuk jangka waktu yang lama. Kedua, PP memiliki ketahanan rendah terhadap suhu rendah, sehingga dapat menjadi rapuh dan mudah retak dalam suhu dingin yang ekstrem. Ketiga, PP sulit untuk didaur ulang secara efisien dan membutuhkan proses khusus untuk daur ulang yang efektif.',
        'logo': 'https://storage.googleapis.com/trashurebucket/dic_ml/Symbol_Resin_Code_5_PP.svg'
    },
    'ps': {
        'kode': 'ps',
        'nama': 'Polystyrene',
        'namalain': 'Polystyrene',
        'manfaat': 'PS (Polystyrene) memiliki beberapa manfaat yang membuatnya banyak digunakan dalam industri kemasan dan aplikasi lainnya. Pertama, PS memiliki sifat isolasi termal yang baik, menjadikannya pilihan yang populer untuk wadah makanan dan minuman yang memerlukan perlindungan terhadap perubahan suhu. Kedua, PS adalah bahan yang ringan dan mudah dibentuk, sehingga digunakan dalam berbagai produk seperti gelas, mangkuk, dan wadah makanan sekali pakai. Ketiga, PS memiliki kekuatan yang baik, memberikan perlindungan yang memadai untuk produk yang dikemas.',
        'kekurangan': 'PS juga memiliki beberapa kekurangan yang perlu dipertimbangkan. Pertama, PS sulit untuk didaur ulang secara efektif dan membutuhkan proses daur ulang yang khusus. Hal ini menyebabkan penumpukan limbah PS yang berkontribusi terhadap masalah limbah plastik. Kedua, pembuangan PS yang tidak tepat dapat menyebabkan pencemaran lingkungan dan berdampak negatif pada ekosistem. Ketiga, PS berpotensi untuk melepaskan bahan kimia berbahaya jika terkena suhu tinggi atau kontak dengan bahan kimia tertentu.',
        'logo': 'https://storage.googleapis.com/trashurebucket/dic_ml/Symbol_Resin_Code_6_PS.svg'
    },
    'other': {
        'kode': 'other',
        'nama': 'other',
        'namalain': '"Other" merupakan istilah umum yang merujuk kepada bahan plastik yang tidak termasuk dalam kategori-kategori yang telah disebutkan sebelumnya. secara umum, manfaat dari bahan plastik dalam kategori ""Other"" dapat meliputi kekuatan mekanis yang baik, kemampuan daur ulang tertentu, keberagaman aplikasi yang luas, dan biaya produksi yang relatif rendah. Namun, kekurangan yang mungkin terkait dengan bahan plastik dalam kategori ""Other"" adalah rentan terhadap kerentanan kimia tertentu, sulitnya daur ulang, dan dampak lingkungan yang mungkin terkait dengan produksi dan pembuangan limbahnya." Penting untuk dicatat bahwa bahan plastik dalam kategori ini dapat memiliki karakteristik yang bervariasi tergantung pada jenis dan komposisi spesifiknya.',
        'manfaat' : '"Other" merupakan istilah umum yang merujuk kepada bahan plastik yang tidak termasuk dalam kategori-kategori yang telah disebutkan sebelumnya. secara umum, manfaat dari bahan plastik dalam kategori "Other" dapat meliputi kekuatan mekanis yang baik, kemampuan daur ulang tertentu, keberagaman aplikasi yang luas, dan biaya produksi yang relatif rendah. Namun, kekurangan yang mungkin terkait dengan bahan plastik dalam kategori "Other" adalah rentan terhadap kerentanan kimia tertentu, sulitnya daur ulang, dan dampak lingkungan yang mungkin terkait dengan produksi dan pembuangan limbahnya.',
        'kekurangan': 'Penting untuk diingat bahwa manfaat dan kekurangan bahan plastik dalam kategori "Other" sangat bergantung pada jenis dan karakteristik spesifik dari masing-masing bahan tersebut. Oleh karena itu, dalam penggunaan bahan plastik dalam kategori "Other," perlu dilakukan penelitian lebih lanjut untuk memahami sifat dan dampaknya secara spesifik serta mempertimbangkan alternatif yang lebih berkelanjutan dan ramah lingkungan jika memungkinkan.',
        'logo': 'https://storage.googleapis.com/trashurebucket/dic_ml/Symbol_Resin_Code_7_OTHER.svg'
    }
}


# Preprocess input image
def preprocess_input(image):
    img_height, img_width = (200, 200)
    image = image.resize((img_height, img_width))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    input_arr = input_arr.astype('float32') / 255
    return input_arr

# Predict image label
def predict_label(image):
    input_arr = preprocess_input(image)
    result = model.predict(input_arr)
    predicted_label = class_names[np.argmax(result)]
    return predicted_label

@app.route('/scan', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is None:
            return jsonify({'error': True, 'message': "file is not found",}), 400
        try:
            file_contents = file.read()
            file_stream = BytesIO(file_contents)

            image = tf.keras.preprocessing.image.load_img(file_stream, target_size=(200, 200))
            predicted_label = predict_label(image)
            
            name = predicted_label
            name = name.split("_")
            res = name[0]
            plastik = plastic_dict[res]            
            # Output JSON
            output = {
                'jenis_sampah': predicted_label,
                'tipe_sampah': plastik['kode'],
                'nama_lain': plastik['namalain'],
                'logo': plastik['logo'],
                'manfaat': plastik['manfaat'],
                'kekurangan': plastik['kekurangan']
            }
            
            return jsonify({
                'error': False,
                'message': "Scan successfull",
                'result': output}), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({"error": True, 'message': 'Method not allowed'}), 405

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 8080))
