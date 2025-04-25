
window.onload() = 

async function translateText() {
    try {
        const textToTranslate = document.getElementById("textToTranslate").value;
        //console.log(textToTranslate) ;
        
        const response = await fetch("http://127.0.0.1:8000/translate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ text: textToTranslate })
        });

        if (!response.ok) {
            throw new Error("Erreur HTTP : ${response.status}");
            
        }
        const data = await response.json();
        document.getElementById("translatedText").innerText = data.translated_text;

    } catch (error) {
        console.error("erreur lors de la traduction : ", error)
    }
    


}

