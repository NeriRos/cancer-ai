document.getElementById('question_form').addEventListener("submit", async function (e) {
    e.preventDefault();
    toggleLoading(true)
    const formData = new FormData(e.target);
    const response = await sendQuestion(formData.get('question'));
    await parseAnswer(await response.json())
    toggleLoading(false)
})

async function sendQuestion(question) {
    return await fetch('/ask', {
        method: "POST",
        body: JSON.stringify({question}),
        headers: {
            'Content-Type': 'application/json'
        }
    });
}

async function parseAnswer(answers) {
    for (const answer of answers) {
        const item = document.createElement('p')
        item.innerText = answer.text;

        document.getElementById('answers__list').appendChild(item)
    }

    document.getElementById('answers').style.display = 'block';
}

function toggleLoading(status) {
    if (status) {
        document.getElementById('loading').style.display = 'block';
        document.getElementById('submit').style.display = 'none';
    } else {
        document.getElementById('submit').style.display = 'block';
        document.getElementById('loading').style.display = 'none';
    }
}

function reset() {
    document.getElementById('answers').style.display = 'none';
    document.getElementById('answers__list').innerHTML = '';
}