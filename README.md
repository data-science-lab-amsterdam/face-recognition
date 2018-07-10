# Face recognition

De gezichtsherkenning werkt als volgt:
* Beeld komt van de webcam
* Een gezichtsdetedctie-algoritme (HOG) herkent waar in het beeld gezichten voorkomen
* Dit deel wordt uitgesneden en is input voor het model
* Het model is een ResNet CNN dat de input vertaalt naar een feature vector van lengte 128
* We hebben een kleine database met de feature vectors van alle DSL medewerkers
* De beste match wordt gevonden dmv een simpele distance metric

Hoe krijg je het aan de praat?
* Maak een environment aan en installeer de requirements. Let op: voordat je dlib kunt installeren moet je eerst e.e.a. al geinstalleerd hebben. Zie: https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/
* git clone de repo
* Plaats in de map ./images van elk persoon één foto
* run main.py
