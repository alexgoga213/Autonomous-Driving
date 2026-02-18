Acest proiect implementează un sistem de detecție a obiectelor bazat pe o arhitectură de tip YOLO (You Only Look Once), dezvoltat de la zero folosind PyTorch. 
Scopul sistemului este de a detecta și clasifica obiecte (ex. indicatoare rutiere / autoturisme) în imagini, prin prezicerea simultană a poziției (bounding box) și a clasei obiectului.
Modelul tratează detecția ca o problemă de regresie, împărțind imaginea într-o grilă și realizând predicții pentru fiecare celulă.

Format: YOLO
Fiecare imagine are un fișier .txt asociat
Format etichete:
class_id x_center y_center width height
(toate valorile sunt normalizate în intervalul [0,1])

Setul de date este împărțit în:
Train – antrenarea modelului
Validation – evaluare intermediară
Test – evaluare finală (date nevăzute)

Performanța este evaluată folosind:
Precision
Recall
F1-score
Criteriul de potrivire este:
IoU ≥ 0.5
Cel mai bun model este salvat pe baza scorului F1 obținut pe setul de validare.

Proiect realizat în scop educațional, pentru studiul detecției de obiecte folosind rețele neuronale convoluționale.
