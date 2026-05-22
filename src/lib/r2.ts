import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";

// Configuración de Cloudflare R2
const accountId = import.meta.env.VITE_R2_ACCOUNT_ID as string;
const accessKeyId = import.meta.env.VITE_R2_ACCESS_KEY_ID as string;
const secretAccessKey = import.meta.env.VITE_R2_SECRET_ACCESS_KEY as string;

export const r2BucketName = import.meta.env.VITE_R2_BUCKET_NAME as string;
export const r2PublicUrl = import.meta.env.VITE_R2_PUBLIC_URL as string;

// Inicializamos el cliente S3 apuntando a Cloudflare R2
export const r2Client = new S3Client({
  region: "auto",
  endpoint: `https://${accountId}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: accessKeyId || "",
    secretAccessKey: secretAccessKey || "",
  },
});

/**
 * Función auxiliar para subir archivos directamente a R2 desde el cliente
 * (Nota de Arquitectura: en producción estricta, usar URLs prefirmadas desde un backend)
 */
export async function uploadFileToR2(file: File, filename: string) {
  if (!accountId || !accessKeyId) {
    throw new Error("Credenciales de R2 no configuradas en el entorno.");
  }

  const command = new PutObjectCommand({
    Bucket: r2BucketName,
    Key: filename,
    Body: file,
    ContentType: file.type,
  });

  await r2Client.send(command);
  
  // Retorna la URL pública si está configurada, o una URL de referencia
  return r2PublicUrl ? `${r2PublicUrl}/${filename}` : `r2://${r2BucketName}/${filename}`;
}
