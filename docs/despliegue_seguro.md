# Despliegue seguro y reproducible

## Separación de responsabilidades

El frontend público en Cloudflare Workers y el backend público en Render deben
mostrar artefactos ya generados. No se recomienda ejecutar los protocolos 5x5 en
un servicio web gratuito: TensorFlow usa CPU durante horas, el proceso puede ser
reiniciado y el filesystem de ese plan no constituye almacenamiento científico
durable.

```text
entorno de entrenamiento controlado
  -> checkpoints + manifiestos + hashes + sellos
  -> revisión y paquete de resultados
  -> backend público de solo lectura
  -> dashboard Cloudflare
```

## Render

`render.yaml` instala dependencias sin descargar o transformar datasets durante
el build. Define health check y genera `API_WRITE_TOKEN`, por lo que las rutas
POST/PUT/PATCH/DELETE requieren `X-API-Key`. El token no debe configurarse como
`VITE_*` ni enviarse al navegador público.

Variables mínimas:

- `API_WRITE_TOKEN`: secreto largo, almacenado solo en Render/cliente administrativo.
- `CORS_ALLOWED_ORIGINS`: dominio exacto de Cloudflare.
- claves de fuentes externas únicamente si el adaptador correspondiente las usa.

Las cabeceras `X-Content-Type-Options: nosniff` y `Referrer-Policy: no-referrer`
se aplican tanto a respuestas exitosas como a rechazos `401`, evitando que la
ruta temprana de autenticacion omita las defensas del resto de la API.

## Cloudflare

El frontend utiliza `VITE_API_URL` para apuntar al backend. Esta variable contiene
una URL, nunca credenciales. El dashboard público puede leer resultados y estados,
pero sus botones de mutación recibirán 401 en producción; las operaciones se
ejecutan localmente o desde un cliente administrativo.

## Publicación de resultados

Antes de publicar una corrida:

1. comprobar que el manifiesto tenga estado candidato de tesis;
2. ejecutar la puerta de congelación con verificación de artefactos;
3. conservar el paquete y sello SHA-256 fuera del filesystem efímero;
4. publicar solo tablas/predicciones necesarias, sin datasets restringidos;
5. verificar que `testSetUsed=false` hasta la autorización final;
6. reconstruir frontend y ejecutar las 108 pruebas del backend.

La fuente principal de Finanzas es ahora el recurso público Índices Soberanos del
MEF. IEEE-CIS permanece solo como adaptador histórico opcional y, si alguna vez se
usa, no puede copiarse al repositorio, imagen o bucket público.

## Ultima verificacion tecnica

El 2 de septiembre de 2026 se aprobaron 108/108 pruebas backend, el verificador
del proyecto, lint y la compilacion cliente/SSR. `npm audit` detecto inicialmente
una vulnerabilidad transitiva de severidad baja en `postcss-selector-parser`; se
actualizo el lockfile con la correccion compatible y la repeticion de la auditoria
reporto cero vulnerabilidades. `pip-audit` reporto cero vulnerabilidades conocidas
tanto para `requirements.txt` como para `requirements-dev.txt`.

## Comandos de control

```powershell
npm audit
backend/.venv/Scripts/python.exe -m pip_audit -r backend/requirements-dev.txt
$env:PYTHONPATH = "backend"
backend/.venv/Scripts/python.exe -m unittest discover -s backend/tests
npm run lint
npm run verify
npm run build
```
