# Face recognition

De gezichtsherkenning werkt als volgt:

![Schema](approach-schema.png)

* Beeld komt van de webcam
* Een gezichtsdetedctie-algoritme (HOG) herkent waar in het beeld gezichten voorkomen
* Dit deel wordt uitgesneden en is input voor het model
* Het model is een ResNet CNN dat de input vertaalt naar een feature vector van lengte 128
* We hebben een kleine database met de feature vectors van alle DSL medewerkers
* De beste match wordt gevonden dmv een simpele distance metric

## Installeren
* Maak een environment aan en installeer de requirements. Let op: voordat je dlib kunt installeren moet je eerst e.e.a. al geinstalleerd hebben. Zie: https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/
* git clone de repo
* Plaats in de map ./images van elk persoon één foto

### Runnen
Je kunt op verschillende
* ```python src/main-webcam-simple.py```
  <br>Toont het beeld in een apart window. Deze is traag en niet aan te bevelen.
  <br>*N.B. Ivm een bug in OpenCV op MacOS sluit het camera window niet!*
* ```python src/main-advanced.py -d -s```
  <br>De ```-d``` zorgt voor een window met beeld
  <br>De ```-s``` zorgt voor geluid
* ```python src/dash-app.py```
  <br>Voor een leuk dashboardje, te zien via http://127.0.0.1:8234/


## RasPi camera op kantoor

Hoe gebruik je de Raspberry Pi met camera die op kantoor hangt? Hier een kleine handleiding:
* Zet de RasPi aan (power-kabel erin, en de stekker :))
* Een opstartscript zorgt ervoor dat de camera gaat draaien en het beeld over het netwerk verzend
* Check op http://192.168.1.163:8554/stream/ of je beeld hebt. Indien niet:
    * Open een terminal en doe `ssh pi:Wodanfabriek23C@192.168.1.163`
    * `./start-picam.sh`
    * Als ie het nu ook niet doet heb jij hem helaas gesloopt...
* run `main-advanced.py -d -s -n` (-n is voor network mode, -d is voor beeld (display) en -s is voor geluid (sound))
