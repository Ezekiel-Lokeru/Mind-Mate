const API_BASE = 'http://127.0.0.1:8000';

function el(id){return document.getElementById(id)}

let chart = null

function renderChart(data){
  // handle empty or missing trend data
  if(!data || !data.trends || Object.keys(data.trends).length === 0){
    if(chart) { chart.destroy(); chart = null }
    // leave the trendsOut text to the caller
    return
  }

  const ctx = el('trendsChart').getContext('2d')
  const labels = Object.keys(data.trends)
  const datasets = labels.map((k,i)=>({
    label: k,
    data: data.trends[k],
    borderColor: `hsl(${(i*60)%360} 80% 60%)`,
    backgroundColor: `hsla(${(i*60)%360} 80% 60% / 0.12)`,
    fill: true,
    tension: 0.3,
  }))

  if(chart) chart.destroy()
  chart = new Chart(ctx, {type:'line', data:{labels: Array( (data.trends[labels[0]]||[]).length ).fill('').map((_,i)=>`D${i+1}`), datasets}})
}

async function postEntry(){
  // By default, submit only free-text to keep UX low-friction.
  // If the user opened Advanced options, include those fields in the payload.
  const payload = { text: el('text').value || '' };
  const adv = document.getElementById('advanced')
  if (adv && adv.open) {
    // include optional advanced fields only when explicitly requested
    const uid = el('user_id').value
    if (uid) payload.user_id = uid
    const scoreVal = parseFloat(el('score').value)
    if (!Number.isNaN(scoreVal)) payload.score = scoreVal
    const tagsVal = (el('tags').value || '').split(',').map(s=>s.trim()).filter(Boolean)
    if (tagsVal.length) payload.tags = tagsVal
  }

  el('response').textContent = 'Sending...'
  try{
    const r = await fetch(`${API_BASE}/entry`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    if(!r.ok){
      const err = await r.json().catch(()=>({detail:'Invalid request'}))
      el('response').textContent = `Error ${r.status}: ${JSON.stringify(err)}`
      showToast('Error submitting entry', 'error')
      return
    }
    const data = await r.json();
    // Render a user-friendly response instead of raw JSON
    let out = ''
    if(data.message){
      out += `<div class="resp-msg">${String(data.message)}</div>`
    }
    if(Array.isArray(data.suggestions) && data.suggestions.length){
      out += '<div class="resp-sugg"><strong>Suggestions:</strong><ul>'
      for(const s of data.suggestions){
        if(typeof s === 'string') out += `<li>${s}</li>`
        else if(s.prompt) out += `<li>${s.prompt}</li>`
        else if(s.type) out += `<li>${s.type}${s.id?`: ${s.id}`:''}</li>`
        else out += `<li>${JSON.stringify(s)}</li>`
      }
      out += '</ul></div>'
    }
    if(data.journal_id){
      out += `<div class="resp-id">Journal ID: <code>${data.journal_id}</code></div>`
    }
    el('response').innerHTML = out || JSON.stringify(data, null, 2)
    showToast(data.message || 'Entry submitted', 'success')
    await getTrends()
  }catch(e){
    el('response').textContent = String(e);
  }
}

async function getTrends(){
  try{
    const r = await fetch(`${API_BASE}/trends`);
    const data = await r.json();
    el('trendsOut').textContent = JSON.stringify(data, null, 2);
    renderChart(data)
  }catch(e){
    el('trendsOut').textContent = String(e);
  }
}

document.addEventListener('DOMContentLoaded', ()=>{
  el('submit').addEventListener('click', postEntry);
  el('refresh').addEventListener('click', getTrends);
  getTrends();
});

// Theme toggle + persistence
const THEME_KEY = 'mm_theme'
function applyTheme(theme){
  if(theme === 'light'){
    document.documentElement.classList.add('theme-light')
    el('themeToggle').setAttribute('aria-pressed','true')
  } else {
    document.documentElement.classList.remove('theme-light')
    el('themeToggle').setAttribute('aria-pressed','false')
  }
  try{ localStorage.setItem(THEME_KEY, theme) }catch(e){}
}
el('themeToggle').addEventListener('click', ()=>{
  const cur = localStorage.getItem(THEME_KEY) || 'dark'
  const next = cur === 'light' ? 'dark' : 'light'
  applyTheme(next)
})
// initialize
(function(){
  const saved = localStorage.getItem(THEME_KEY) || 'dark'
  applyTheme(saved)
})();

// Toast helpers
function showToast(message, type='success', timeout=4000){
  const c = document.getElementById('toast-container')
  if(!c) return
  const t = document.createElement('div')
  t.className = `toast ${type}`
  t.setAttribute('role','status')
  t.textContent = message
  c.appendChild(t)
  setTimeout(()=>{ t.style.opacity = '0'; t.style.transform='translateY(12px)'; setTimeout(()=>t.remove(),300) }, timeout)
}

