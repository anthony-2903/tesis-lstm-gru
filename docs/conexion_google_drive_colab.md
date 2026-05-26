# Conexion del frontend con Google Drive para Colab

## Objetivo

Permitir que el frontend suba el dataset a una carpeta de Google Drive. Luego
Google Colab lee ese archivo desde Drive y ejecuta el entrenamiento.

```text
Frontend Vercel/local
  -> usuario sube CSV/XLSX
  -> Google OAuth autoriza Drive
  -> archivo se copia a carpeta TESIS_COLAB/backend
  -> Colab monta Drive y entrena modelos
```

## Variables necesarias

Agrega estas variables a `.env.local` para desarrollo y a Vercel para produccion:

```env
VITE_GOOGLE_CLIENT_ID=tu_client_id.apps.googleusercontent.com
VITE_GOOGLE_DRIVE_FOLDER_ID=id_de_la_carpeta_drive
```

## Obtener `VITE_GOOGLE_DRIVE_FOLDER_ID`

1. Abre Google Drive.
2. Entra a la carpeta donde quieres guardar los datasets, por ejemplo:

```text
Mi unidad/TESIS_COLAB/backend
```

3. Copia el ID desde la URL:

```text
https://drive.google.com/drive/folders/ESTE_ES_EL_FOLDER_ID
```

Ese valor va en:

```env
VITE_GOOGLE_DRIVE_FOLDER_ID=ESTE_ES_EL_FOLDER_ID
```

## Obtener `VITE_GOOGLE_CLIENT_ID`

1. Entra a Google Cloud Console.
2. Crea un proyecto o usa uno existente.
3. Habilita la API:

```text
Google Drive API
```

4. Configura la pantalla de consentimiento OAuth.
5. Crea una credencial:

```text
OAuth client ID -> Web application
```

6. En origenes JavaScript autorizados agrega:

```text
http://localhost:5173
https://tu-dominio-vercel.vercel.app
```

7. Copia el Client ID generado y usalo en:

```env
VITE_GOOGLE_CLIENT_ID=tu_client_id.apps.googleusercontent.com
```

## Flujo de uso

1. Inicia el frontend.
2. Ve a `Subir Datos`.
3. Presiona `Conectar Drive`.
4. Autoriza con tu cuenta Google.
5. Sube el CSV/XLSX.
6. El archivo se procesara en el dashboard y tambien se copiara a Drive.
7. En Colab ejecuta:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/TESIS_COLAB/backend
!python colab_training_pipeline.py --dataset datos.csv --target ULT_PROBLEMA
```

Si el archivo subido queda con timestamp, revisa el nombre con:

```python
!ls
```

Y usa ese nombre en `--dataset`.

## Nota tecnica

La subida usa Google Identity Services para obtener un token OAuth en el navegador
y Google Drive API con `uploadType=resumable`, adecuado para archivos CSV/XLSX
grandes. El scope usado es:

```text
https://www.googleapis.com/auth/drive.file
```

Esto permite crear archivos en Drive con autorizacion del usuario.
