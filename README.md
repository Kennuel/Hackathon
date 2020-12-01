# ITF Hackathon 11. Dezember 2020  
## Aufgabe
In der vorliegenden Aufgabe soll ein Datenset bearbeitet werden, dass aus den Sensordaten einer Röstmaschine und der daraus resultierenden Produktqualität darstellt.

Das Datenset kann unter folgen Link gecloned werden: < GITLAB LINK ZUM DATENSET >.

Ein Datensatz enthält dabei 18 Merkmale (Features). Sowie das Zieldatum: Quality - Die Daten finden sich in der data_X.csv:
date_time, T_data_1_1, T_data_1_2, T_data_1_3, T_data_2_1, T_data_2_2, T_data_2_3, T_data_3_1, T_data_3_2, T_data_3_3, T_data_4_1 ,T_data_4_2, T_data_4_3, T_data_5_1, T_data_5_2, T_data_5_3, H_data, AH_data, quality

**date_time** ist der Zeitpunkt des Beginns des Röstprozesses
**T_data_X_Y** ist der Temperatursensor Y in einer der fünf Röstkammern X
**H_data** Höhe des Produkts vor dem Rösten
**AH_data** Feuchtigkeitsgehalt des Produkts vor dem Rösten
**quality** Die Qualität des Produkts

In der data_Y.csv findet sich die Qualität und der Timestamp der Prüfung. Die Werte reichen von 221 bis 505.

Ziel ist es mit Hilfe eines neuronalen Netzwerks eine möglichst genaue Vorhersage der Qualität aus den übergebenen Features zu berechnen.

Aus dem Testset wurden einige Daten entnommen. Diese werden am Ende genutzt, um das Endergebnis des Hackathons zu bestimmen. 

Dabei wird der Fehler über Folgenden Funktion bestimmt:
<img src="a2f90a7d72270a9d6a54e6671d0f7a16.png" alt="MAE" width="220"/>

Es wird die Summe über alle Abweichungen zwischen dem erwarteten und dem berechneten Output berechnet und daraus der Durchschnitt gemeldet. Es handelt sich dabei um die mittlere absolute Abweichung oder mean absolute error.

### Ziel
Es muss am Ende eine (Python)funktion existieren, in der wir einen Pfad zu einer CSV Datei angeben können. Diese CSV Datei wird das gleiche Format haben wie die oben angegebene. Die Funktion muss diese CSV laden und auf Basis ihrer Inhalte predictions auf einem vorher trainierten Netz oder Model gemacht werden.
Am Ende muss diese Funktion den MAE zwischen der Prediction und der Qualityspalte angeben.

PSEUDOCODE:
``` 
function() {
    csv := loadCSV(path)
    MAE := 0
    model := load_model(modelPath)
    preprocessedData := preprocessPipeline(csv)
    for(row in preprocessedData) {
        MAE = abs( model.predict(row) - row.quality)
    }
    MAE = MAE / preprocessedData.length
    print(MAE)

}
```

## How to get started
Der erste Schritt sollte es sein das oben genannte Repository zu klonen und sich die Daten einmal in einem Texteditor anzuschauen.

Zur Bearbeitung bietet sich ein Google-Collab-Notebook an. Dieses kann unter folgendem Link erstellt werden: https://colab.research.google.com/
Es bietet sich an den Laufzeittyp auf GPU zu ändern. Dafür im Menü Laufzeit -> Laufzeittyp ändern auswählen.
Die .csv-Dateien können in das Google-Collab-Notebook hochgeladen werden.

Wichtige Bibliotheken, die importiert werden sollten sind numpy und pandas.

```python
import numpy as np 
import pandas as pd
```

Über pandas können die Daten geladen und analysiert werden:
```python
train_X = pd.read_csv('data_X.csv')
train_X.describe()
```

Als Frameworks für das neuronale Netzwerk bieten sich besonders Keras und Ludwig an.

Für Keras gibt es hier ein Notebook, das einen guten ersten Anhaltspunkt bietet: https://www.kaggle.com/podsyp/product-quality-with-keras

Für Ludwig ist die erstklassige Dokumentation zu empfehlen: 
https://ludwig-ai.github.io/ludwig-docs/?from=%40.
Auch dieser Blogartikel bietet einen ersten guten Überblick:
https://gilberttanner.com/blog/introduction-to-ubers-ludwig

## Was kann man alles probieren?
Im folgenden ein paar Dinge, die man ausprobieren könnte, um bessere Ergebnisse zu erzielen:
1. Train/Test-Split anpassen
2. Netzwerk vergrößern
3. Netzwerk verkleinern
4. Aktivierungsfunktionen verändern
5. Optimizer verändern
6. Dropout-Layer einfügen
7. Länger trainieren
8. Kürzer trainieren
9. Daten normalisieren
10. Extremwerte aus den Trainingsdaten entfernen
11. Initialisierung der Werte verändern
12. Lossfunktion verändern





