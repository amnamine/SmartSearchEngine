const searchInput = document.getElementById('searchInput');
const resultsGrid = document.getElementById('resultsGrid');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsHeader = document.getElementById('resultsHeader');
const resultsCount = document.getElementById('resultsCount');

// Allow Enter key to search
searchInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') {
        performSearch();
    }
});

function fillSearch(text) {
    searchInput.value = text;
    performSearch();
}

async function performSearch() {
    const query = searchInput.value.trim();
    if (!query) return;

    // UI State: Loading
    resultsGrid.innerHTML = '';
    loadingSpinner.classList.remove('hidden');
    resultsHeader.classList.add('hidden');

    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query }),
        });

        const results = await response.json();
        renderResults(results, query);
    } catch (error) {
        console.error('Error:', error);
        resultsGrid.innerHTML = '<p class="empty-state">Erreur de connexion au serveur.</p>';
    } finally {
        loadingSpinner.classList.add('hidden');
    }
}

function renderResults(results, query) {
    if (results.length === 0) {
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-search-minus"></i>
                <p>Aucun résultat trouvé pour "${query}".</p>
            </div>
        `;
        return;
    }

    resultsHeader.classList.remove('hidden');
    resultsCount.textContent = `${results.length} produits trouvés`;

    results.forEach(product => {
        const card = document.createElement('div');
        card.className = 'card';

        // Format image: if it fails, placeholder is handled in app.py, but we can add onerror here too
        // Format price: Add space for thousands if needed (simple implementation)

        card.innerHTML = `
            <img src="${product.image}" alt="${product.name}" class="card-img" onerror="this.src='https://via.placeholder.com/200x200?text=Smart+Search'">
            <div class="card-body">
                <span class="card-cat">${product.category}</span>
                <h3 class="card-title">${product.name}</h3>
                <div class="ai-match">
                    <i class="fas fa-robot"></i> Match ${product.score}%
                </div>
                <span class="card-price">${product.price}</span>
                <a href="#" class="btn-details">Voir détails</a>
            </div>
        `;
        resultsGrid.appendChild(card);
    });
}