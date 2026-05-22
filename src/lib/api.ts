// Funciones para interactuar con el backend FastAPI

const API_URL = "http://localhost:8000/api";

export async function fetchDashboardData() {
  const response = await fetch(`${API_URL}/dashboard`);
  if (!response.ok) {
    throw new Error("Error fetching dashboard data");
  }
  return response.json();
}
