// Dynamic Dashboard Logic
// Author: Aman Kumar

document.addEventListener('DOMContentLoaded', () => {
    fetchStats();
    fetchInvoices();
    setupNavigation();
    setupUpload();
});

// --- State ---
let riskChart = null;

// --- API Calls ---
async function fetchStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        
        document.getElementById('stat-total').innerText = data.total;
        document.getElementById('stat-critical').innerText = data.critical;
        document.getElementById('stat-high').innerText = data.high;
        document.getElementById('stat-value').innerText = data.total_value;
        
        updateCharts(data);
    } catch (err) {
        console.error('Failed to fetch stats', err);
    }
}

async function fetchInvoices() {
    try {
        const res = await fetch('/api/invoices');
        const data = await res.json();
        renderTable(data);
    } catch (err) {
        console.error('Failed to fetch invoices', err);
    }
}

// --- UI Rendering ---
function renderTable(invoices) {
    const tbody = document.getElementById('invoice-tbody');
    tbody.innerHTML = '';

    if (invoices.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; padding:40px;">No flagged invoices found.</td></tr>';
        return;
    }

    invoices.forEach(inv => {
        const tr = document.createElement('tr');
        const score = inv.final_risk_score;
        const color = getScoreColor(score);
        
        tr.innerHTML = `
            <td><strong>${inv.invoice_number}</strong></td>
            <td>${inv.vendor_name || inv.vendor_id}</td>
            <td>₹${inv.invoice_amount.toLocaleString()}</td>
            <td>
                <div class="score-cell">
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width: ${score}%; background-color: ${color}"></div>
                    </div>
                    <span>${score.toFixed(1)}</span>
                </div>
            </td>
            <td><span class="risk-badge badge-${inv.risk_level.toLowerCase()}">${inv.risk_level}</span></td>
            <td><button class="btn-outline" onclick="viewDetail('${inv.invoice_number}')">View Details</button></td>
        `;
        tbody.appendChild(tr);
    });
}

function getScoreColor(score) {
    if (score >= 75) return '#ef4444';
    if (score >= 55) return '#f97316';
    if (score >= 35) return '#eab308';
    return '#22c55e';
}

function updateCharts(stats) {
    const ctx = document.getElementById('riskChart').getContext('2d');
    
    if (riskChart) riskChart.destroy();
    
    riskChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Critical', 'High', 'Medium', 'Safe'],
            datasets: [{
                data: [stats.critical, stats.high, (stats.total * 0.1), (stats.total * 0.8)],
                backgroundColor: ['#ef4444', '#f97316', '#eab308', '#22c55e'],
                borderWidth: 0
            }]
        },
        options: {
            plugins: { legend: { position: 'bottom', labels: { color: '#9ca3af' } } },
            cutout: '70%'
        }
    });
}

// --- Navigation ---
function setupNavigation() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const viewId = item.getAttribute('data-view');
            
            // Update nav state
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            
            // Update view
            document.querySelectorAll('.view').forEach(v => v.classList.add('hidden'));
            document.getElementById(`view-${viewId}`).classList.remove('hidden');
        });
    });
}

// --- File Upload ---
function setupUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.querySelector('.browse');

    browseBtn.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });
}

function handleFiles(files) {
    if (files.length === 0) return;
    const file = files[0];
    
    const info = document.getElementById('file-info');
    document.getElementById('filename').innerText = file.name;
    info.classList.remove('hidden');

    document.getElementById('btn-analyze').onclick = () => uploadFile(file);
}

async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const btn = document.getElementById('btn-analyze');
    btn.innerText = 'Analyzing...';
    btn.disabled = true;

    try {
        const res = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await res.json();
        
        if (data.status === 'success') {
            alert(`Analysis Complete! Processed ${data.count} invoices and flagged ${data.flagged}.`);
            window.location.reload();
        } else {
            alert('Error: ' + data.error);
        }
    } catch (err) {
        alert('Upload failed');
    } finally {
        btn.innerText = 'Start AI Analysis';
        btn.disabled = false;
    }
}

// --- Detail View ---
async function viewDetail(invNum) {
    const res = await fetch(`/api/explain/${invNum}`);
    const data = await res.json();
    
    const modal = document.getElementById('detail-modal');
    const body = document.getElementById('modal-body');
    
    body.innerHTML = `
        <h2 style="margin-bottom: 20px;">Analysis for ${invNum}</h2>
        <div style="background:#0a0e17; padding:20px; border-radius:10px; font-family:monospace; white-space:pre-wrap; font-size:13px; line-height:1.5;">
            ${data.explanation}
        </div>
        <div style="margin-top: 20px; display:flex; gap:10px;">
             <button class="btn-primary" onclick="closeModal()">Dismiss</button>
             <button class="btn-outline">Flag as False Positive</button>
        </div>
    `;
    
    modal.classList.remove('hidden');
}

function closeModal() {
    document.getElementById('detail-modal').classList.add('hidden');
}

function showUploadModal() {
    document.querySelector('[data-view="upload"]').click();
}
