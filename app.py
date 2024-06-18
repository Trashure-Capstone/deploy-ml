from flask import Flask, request, jsonify
import numpy as np
import os
from io import BytesIO
import onnxruntime
import cv2

app = Flask(__name__)

# Create an InferenceSession instance to load the model
ort_session = onnxruntime.InferenceSession('best.onnx')

class_names = ["HDPE", "LDPE", "PET", "PP", "PS"]

plastic_dict = {
    'HDPE': {
        'kode': 'hdpe',
        'nama': 'High density polyethylene',
        'manfaat': 'manfaat HDPE (High-Density Polyethylene) sebagai bahan plastik termasuk ketahanan terhadap bahan kimia, seperti asam dan alkali, serta tahan terhadap kelembaban dan air. HDPE juga dapat didaur ulang untuk mengurangi limbah plastik dan memiliki kekuatan tarik yang tinggi, membuatnya cocok untuk aplikasi yang membutuhkan kekuatan dan keandalan.',
        'kekurangan': 'HDPE juga memiliki beberapa kekurangan. Misalnya, HDPE rentan terhadap suhu tinggi dan dapat deformasi atau meleleh jika terpapar suhu yang ekstrem. Selain itu, HDPE tidak transparan, sehingga tidak cocok untuk produk yang memerlukan transparansi atau tampilan visual. HDPE juga tidak tahan terhadap panas dan dapat retak jika terkena tekanan atau gaya bengkok yang ekstrem.',
        'logo': 'https://upload.wikimedia.org/wikipedia/commons/0/01/Symbol_Resin_Code_2_HDPE.svg'
    },
    'LDPE': {
        'kode': 'ldpe',
        'nama': 'Low density polyethylene',
        'manfaat': 'LDPE (Low-Density Polyethylene) memiliki beberapa manfaat yang membuatnya populer dalam berbagai aplikasi. Pertama, LDPE memiliki sifat tahan terhadap bahan kimia, seperti asam dan alkali, menjadikannya pilihan yang baik untuk produk-produk yang memerlukan ketahanan terhadap korosi. Kedua, LDPE memiliki sifat tahan terhadap kelembaban dan air, sehingga sering digunakan dalam produk-produk yang terpapar kelembaban tinggi. Ketiga, LDPE memiliki kekuatan yang baik dan tahan terhadap tekanan, menjadikannya pilihan yang baik untuk produk-produk yang membutuhkan kekuatan mekanis yang tinggi.',
        'kekurangan': 'LDPE juga memiliki beberapa kekurangan yang perlu dipertimbangkan. Pertama, LDPE rentan terhadap suhu tinggi dan dapat meleleh atau deformasi jika terpapar suhu yang ekstrem. Kedua, LDPE memiliki ketahanan rendah terhadap sinar UV dan dapat mengalami degradasi jika terpapar langsung sinar matahari. Ketiga, LDPE sulit untuk didaur ulang secara efisien dan membutuhkan proses khusus untuk daur ulang yang efektif.',
        'logo': 'https://upload.wikimedia.org/wikipedia/commons/0/00/Symbol_Resin_Code_4_LDPE.svg'
    },
    'PET': {
        'kode': 'pet',
        'nama': 'Polyethylene terephthalate',
        'manfaat': 'PET (Polyethylene Terephthalate) memiliki beberapa manfaat yang membuatnya populer dalam berbagai aplikasi. Pertama, PET memiliki sifat tahan terhadap bahan kimia, seperti asam dan alkali, menjadikannya pilihan yang baik untuk produk-produk yang memerlukan ketahanan terhadap korosi. Kedua, PET memiliki sifat tahan terhadap kelembaban dan air, sehingga sering digunakan dalam produk-produk yang terpapar kelembaban tinggi. Ketiga, PET memiliki kekuatan yang baik dan tahan terhadap tekanan, menjadikannya pilihan yang baik untuk produk-produk yang membutuhkan kekuatan mekanis yang tinggi.',
        'kekurangan': 'PET juga memiliki beberapa kekurangan yang perlu dipertimbangkan. Pertama, PET rentan terhadap suhu tinggi dan dapat meleleh atau deformasi jika terpapar suhu yang ekstrem. Kedua, PET memiliki ketahanan rendah terhadap sinar UV dan dapat mengalami degradasi jika terpapar langsung sinar matahari. Ketiga, PET sulit untuk didaur ulang secara efisien dan membutuhkan proses khusus untuk daur ulang yang efektif.',
        'logo': 'https://en.m.wikipedia.org/wiki/File:Symbol_Resin_Code_4_LDPE.svg#/media/File%3APlastic-recyc-01.svg'
    },
    'PP': {
        'kode': 'pp',
        'nama': 'Polypropylene',
        'manfaat': 'PP (Polypropylene) memiliki beberapa manfaat yang membuatnya populer dalam berbagai aplikasi. Pertama, PP memiliki kekuatan yang tinggi dan tahan terhadap tekanan dan benturan, menjadikannya pilihan yang baik untuk produk-produk yang membutuhkan kekuatan mekanis yang tinggi. Kedua, PP memiliki sifat tahan terhadap bahan kimia, seperti asam dan alkali, sehingga sering digunakan dalam industri kimia. Ketiga, PP memiliki sifat tahan terhadap panas dan memiliki titik leleh yang tinggi, sehingga cocok untuk penggunaan dalam produk-produk yang terekspos suhu tinggi.',
        'kekurangan': 'PP juga memiliki beberapa kekurangan yang perlu dipertimbangkan. Pertama, PP rentan terhadap paparan sinar UV dan dapat mengalami degradasi jika terpapar langsung sinar matahari untuk jangka waktu yang lama. Kedua, PP memiliki ketahanan rendah terhadap suhu rendah, sehingga dapat menjadi rapuh dan mudah retak dalam suhu dingin yang ekstrem. Ketiga, PP sulit untuk didaur ulang secara efisien dan membutuhkan proses khusus untuk daur ulang yang efektif.',
        'logo': 'https://en.m.wikipedia.org/wiki/File:Symbol_Resin_Code_4_LDPE.svg#/media/File%3APlastic-recyc-05.svg'
    },
    'PS': {
        'kode': 'ps',
        'nama': 'Polystyrene',
        'manfaat': 'PS (Polystyrene) memiliki beberapa manfaat yang membuatnya banyak digunakan dalam industri kemasan dan aplikasi lainnya. Pertama, PS memiliki sifat isolasi termal yang baik, menjadikannya pilihan yang populer untuk wadah makanan dan minuman yang memerlukan perlindungan terhadap perubahan suhu. Kedua, PS adalah bahan yang ringan dan mudah dibentuk, sehingga digunakan dalam berbagai produk seperti gelas, mangkuk, dan wadah makanan sekali pakai. Ketiga, PS memiliki kekuatan yang baik, memberikan perlindungan yang memadai untuk produk yang dikemas.',
        'kekurangan': 'PS juga memiliki beberapa kekurangan yang perlu dipertimbangkan. Pertama, PS sulit untuk didaur ulang secara efektif dan membutuhkan proses daur ulang yang khusus. Hal ini menyebabkan penumpukan limbah PS yang berkontribusi terhadap masalah limbah plastik. Kedua, pembuangan PS yang tidak tepat dapat menyebabkan pencemaran lingkungan dan berdampak negatif pada ekosistem. Ketiga, PS berpotensi untuk melepaskan bahan kimia berbahaya jika terkena suhu tinggi atau kontak dengan bahan kimia tertentu.',
        'logo': 'https://en.m.wikipedia.org/wiki/File:Symbol_Resin_Code_4_LDPE.svg#/media/File%3APlastic-recyc-06.svg'
    }
}


# Preprocess input image
def preprocess_input(img):
    img = cv2.resize(img, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0
    return img


# Predict image label
def predict_label(img):
    img = preprocess_input(img)
    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs

@app.route('/scan', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file is None:
            return jsonify({'error': True, 'message': "file is not found",}), 400
        try:
            file_contents = file.read()
            file_stream = BytesIO(file_contents)

            image = cv2.imdecode(np.frombuffer(file_stream.read(), np.uint8), cv2.IMREAD_COLOR)
            predicted_label = predict_label(image)
            
            max_prob_index = predicted_label[0].argmax()
            max_prob_class = class_names[max_prob_index]
            full_info = plastic_dict[max_prob_class]
            output = {
                'jenis_sampah': max_prob_class,
                'nama_lain': full_info["nama"],
                'logo': full_info['logo'],
                'manfaat': full_info['manfaat'],
                'kekurangan': full_info['kekurangan']
            }
            
            return jsonify({
                'error': False,
                'message': "Scan successfull",
                'result': output}), 200
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({"error": True, 'message': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=os.environ.get('PORT', 8080), debug=True)
