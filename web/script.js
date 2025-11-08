const input = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const result = document.getElementById("result");

input.addEventListener("change", () => {
  const file = input.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
      preview.src = e.target.result;
      preview.classList.remove("d-none");
    };
    reader.readAsDataURL(file);
  }
});

document.getElementById("btn").addEventListener("click", async () => {
  if (!input.files.length) {
    alert("Choisis une image.");
    return;
  }

  const formData = new FormData();
  formData.append("file", input.files[0]);

  result.textContent = "⏳ Prédiction en cours...";
  result.classList.remove("text-danger", "text-success");

  try {
    const resp = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      body: formData
    });
    const data = await resp.json();
    result.textContent = `✅ Classe prédite : ${data.predicted_class}`;
    result.classList.add("text-success");
  } catch (err) {
    result.textContent = "❌ Erreur lors de la prédiction.";
    result.classList.add("text-danger");
    console.error(err);
  }
});
