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
        const src = product.source || 'boutique';
        const badgeClass = src === 'catalogue' ? 'catalogue' : 'boutique';
        const label = product.source_label || (src === 'catalogue' ? 'Catalogue' : 'Boutique');
        const rawUrl = (product.url || '').trim();
        const safeUrl = rawUrl.replace(/"/g, '&quot;');
        const isBing = /^https:\/\/www\.bing\.com\/search/i.test(rawUrl);
        const linkLabel = isBing ? 'Ouvrir la recherche GSMArena' : 'Fiche GSMArena';
        const linkHtml = rawUrl
            ? `<a href="${safeUrl}" class="btn-details btn-external" target="_blank" rel="noopener noreferrer">${linkLabel}</a>`
            : `<span class="btn-details disabled">Lien non disponible</span>`;

        card.innerHTML = `
            <img src="${product.image}" alt="${product.name}" class="card-img" onerror="this.src='https://via.placeholder.com/200x200?text=Smart+Search'">
            <div class="card-body">
                <span class="source-badge ${badgeClass}">${label}</span>
                <span class="card-cat">${product.category}</span>
                <h3 class="card-title">${product.name}</h3>
                <div class="ai-match">
                    <i class="fas fa-robot"></i> Match ${product.score}%
                </div>
                <span class="card-price">${product.price}</span>
                ${linkHtml}
            </div>
        `;
        resultsGrid.appendChild(card);
    });
}