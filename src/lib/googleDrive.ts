type GoogleTokenResponse = {
  access_token?: string;
  error?: string;
};

type GoogleTokenClient = {
  requestAccessToken: (options?: { prompt?: string }) => void;
  callback?: (response: GoogleTokenResponse) => void;
};

declare global {
  interface Window {
    google?: {
      accounts: {
        oauth2: {
          initTokenClient: (config: {
            client_id: string;
            scope: string;
            callback: (response: GoogleTokenResponse) => void;
          }) => GoogleTokenClient;
        };
      };
    };
  }
}

const GOOGLE_IDENTITY_SCRIPT = "https://accounts.google.com/gsi/client";
const DRIVE_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable";
const DRIVE_FILE_URL = "https://www.googleapis.com/drive/v3/files";
const DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.file";

const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID as string | undefined;
const folderId = import.meta.env.VITE_GOOGLE_DRIVE_FOLDER_ID as string | undefined;

let tokenClient: GoogleTokenClient | null = null;
let accessToken: string | null = null;

export const isGoogleDriveConfigured = Boolean(clientId && folderId);

function loadGoogleIdentityScript() {
  return new Promise<void>((resolve, reject) => {
    if (window.google?.accounts?.oauth2) {
      resolve();
      return;
    }

    const existing = document.querySelector<HTMLScriptElement>(`script[src="${GOOGLE_IDENTITY_SCRIPT}"]`);
    if (existing) {
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", () => reject(new Error("No se pudo cargar Google Identity Services.")), { once: true });
      return;
    }

    const script = document.createElement("script");
    script.src = GOOGLE_IDENTITY_SCRIPT;
    script.async = true;
    script.defer = true;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error("No se pudo cargar Google Identity Services."));
    document.head.appendChild(script);
  });
}

export async function requestGoogleDriveAccess() {
  if (!clientId) {
    throw new Error("Falta VITE_GOOGLE_CLIENT_ID en el entorno.");
  }

  await loadGoogleIdentityScript();

  return new Promise<string>((resolve, reject) => {
    tokenClient = window.google!.accounts.oauth2.initTokenClient({
      client_id: clientId,
      scope: DRIVE_SCOPE,
      callback: (response) => {
        if (response.error || !response.access_token) {
          reject(new Error("Google no devolvio un token de acceso."));
          return;
        }
        accessToken = response.access_token;
        resolve(accessToken);
      },
    });

    tokenClient.requestAccessToken({ prompt: accessToken ? "" : "consent" });
  });
}

async function getAccessToken() {
  if (accessToken) return accessToken;
  return requestGoogleDriveAccess();
}

export async function uploadFileToGoogleDrive(file: File, filename: string) {
  if (!folderId) {
    throw new Error("Falta VITE_GOOGLE_DRIVE_FOLDER_ID en el entorno.");
  }

  const token = await getAccessToken();
  const metadata = {
    name: filename,
    parents: [folderId],
    mimeType: file.type || "application/octet-stream",
  };

  const session = await fetch(DRIVE_UPLOAD_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json; charset=UTF-8",
      "X-Upload-Content-Type": file.type || "application/octet-stream",
      "X-Upload-Content-Length": String(file.size),
    },
    body: JSON.stringify(metadata),
  });

  if (!session.ok) {
    throw new Error(`No se pudo iniciar la subida a Drive (${session.status}).`);
  }

  const uploadUrl = session.headers.get("Location");
  if (!uploadUrl) {
    throw new Error("Google Drive no devolvio una URL de subida.");
  }

  const upload = await fetch(uploadUrl, {
    method: "PUT",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "Content-Length": String(file.size),
    },
    body: file,
  });

  if (!upload.ok) {
    throw new Error(`No se pudo subir el archivo a Drive (${upload.status}).`);
  }

  const created = await upload.json() as { id: string; name: string };
  const details = await fetch(`${DRIVE_FILE_URL}/${created.id}?fields=id,name,webViewLink`, {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!details.ok) {
    return { ...created, webViewLink: `https://drive.google.com/file/d/${created.id}/view` };
  }

  return await details.json() as { id: string; name: string; webViewLink?: string };
}
