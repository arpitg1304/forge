"""Forge Studio HTML template (design system adapted from the Studio mockup).

`generate_studio` injects real catalog JSON at the `/*__DATA__*/` and
`/*__THUMBS__*/` placeholders and the catalog name at `__CATALOG__`.
"""

from __future__ import annotations

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Forge Studio — __CATALOG__</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;450;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{
  --canvas:#0c0f16;--surface:#12161f;--raised:#171c28;--overlay:#1d2433;
  --hairline:rgba(151,166,199,.09);--line:rgba(151,166,199,.15);--line-hi:rgba(151,166,199,.28);
  --edge-light:inset 0 1px 0 rgba(255,255,255,.035);--shadow-1:0 1px 2px rgba(0,0,0,.3);
  --shadow-2:0 4px 16px rgba(0,0,0,.35),0 1px 3px rgba(0,0,0,.4);
  --ink:#eef1f8;--ink-2:#99a3b8;--ink-3:#5c667c;
  --cyan:#4ad8e8;--cyan-soft:rgba(74,216,232,.12);--cyan-line:rgba(74,216,232,.3);--cyan-dim:#1f6b7a;
  --violet:#a78bfa;--violet-soft:rgba(167,139,250,.12);--violet-line:rgba(167,139,250,.3);
  --amber:#f6a72b;--amber-soft:rgba(246,167,43,.12);--amber-line:rgba(246,167,43,.35);--amber-hover:#ffb945;
  --good:#41d693;--mid:#f2c94c;--bad:#f2685c;
  --good-soft:rgba(65,214,147,.12);--mid-soft:rgba(242,201,76,.1);--bad-soft:rgba(242,104,92,.1);
  --disp:'Space Grotesk',system-ui,sans-serif;--body:'IBM Plex Sans',system-ui,sans-serif;
  --mono:'IBM Plex Mono',ui-monospace,'SF Mono',monospace;--r-lg:14px;--ease:cubic-bezier(.25,.7,.3,1);
}
*{box-sizing:border-box;margin:0;padding:0}html,body{height:100%}
body{background:var(--canvas);color:var(--ink);font-family:var(--body);font-size:13.5px;line-height:1.55;-webkit-font-smoothing:antialiased;overflow:hidden}
::selection{background:rgba(74,216,232,.25)}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--overlay);border-radius:8px;border:2px solid var(--canvas)}
:focus-visible{outline:2px solid var(--cyan);outline-offset:2px;border-radius:4px}
.num{font-variant-numeric:tabular-nums}
#app{display:grid;grid-template-columns:206px 1fr;grid-template-rows:52px 1fr;height:100vh}
header{grid-column:1/3;display:flex;align-items:center;gap:14px;padding:0 18px;background:var(--surface);border-bottom:1px solid var(--hairline)}
.logo{display:flex;align-items:center;gap:10px}
.logo-mark{width:26px;height:26px;border-radius:7px;display:grid;place-items:center;background:linear-gradient(140deg,#2f8898,#123340);box-shadow:var(--edge-light);font-family:var(--disp);font-weight:700;color:var(--cyan);font-size:14px}
.logo-name{font-family:var(--disp);font-weight:600;font-size:14.5px}.logo-name span{color:var(--ink-3);font-weight:500}
.ws{display:flex;align-items:center;gap:7px;font-size:12.5px;color:var(--ink-2);padding:5px 11px;border-radius:8px;border:1px solid var(--hairline)}
.ws b{color:var(--ink);font-weight:500;font-family:var(--mono);font-size:11.5px}
.env{margin-left:auto;font-family:var(--mono);font-size:10px;letter-spacing:.08em;color:var(--good);background:var(--good-soft);border:1px solid rgba(65,214,147,.25);border-radius:6px;padding:3px 8px}
nav{background:var(--surface);border-right:1px solid var(--hairline);display:flex;flex-direction:column;padding:14px 10px}
.nav-sec{font-family:var(--mono);font-size:9.5px;letter-spacing:.16em;color:var(--ink-3);text-transform:uppercase;padding:14px 10px 6px}
.nav-item{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:8px;color:var(--ink-2);cursor:pointer;font-size:13px;user-select:none;position:relative;transition:background .12s,color .12s;margin-bottom:1px}
.nav-item svg{width:15px;height:15px;stroke:currentColor;fill:none;stroke-width:1.6;flex-shrink:0}
.nav-item:hover{color:var(--ink);background:rgba(151,166,199,.05)}
.nav-item.active{color:var(--ink);background:var(--raised);box-shadow:var(--edge-light),var(--shadow-1)}
.nav-item.active::before{content:'';position:absolute;left:-10px;top:7px;bottom:7px;width:2px;border-radius:2px;background:var(--amber)}
.nav-item .count{margin-left:auto;font-family:var(--mono);font-size:10px;color:var(--ink-3)}
.nav-item .count.hot{color:var(--amber)}
.nav-foot{margin-top:auto;padding:12px 10px 4px;border-top:1px solid var(--hairline);font-family:var(--mono);font-size:10px;color:var(--ink-3);line-height:1.9}
.nav-foot b{color:var(--ink-2);font-weight:500}
main{overflow-y:auto;padding:26px 30px 60px}
.view{display:none;max-width:1280px;margin:0 auto}.view.active{display:block;animation:viewIn .28s var(--ease)}
@keyframes viewIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
.crumb{font-family:var(--mono);font-size:10.5px;color:var(--ink-3);margin-bottom:6px}.crumb b{color:var(--ink-2);font-weight:500}
h1{font-family:var(--disp);font-weight:600;font-size:23px;letter-spacing:-.015em;margin-bottom:3px}
.sub{color:var(--ink-2);font-size:12.5px;margin-bottom:22px}
h2{font-family:var(--disp);font-weight:600;font-size:13px}
.h2row{display:flex;align-items:baseline;justify-content:space-between;margin-bottom:12px}
.h2row .meta{font-family:var(--mono);font-size:10.5px;color:var(--ink-3)}
.card{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--r-lg);padding:18px;box-shadow:var(--edge-light);margin-bottom:16px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:16px}
.kpi{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--r-lg);padding:15px 17px 13px;box-shadow:var(--edge-light)}
.kpi .lab{font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3)}
.kpi .val{font-family:var(--disp);font-weight:600;font-size:26px;margin:5px 0 1px;letter-spacing:-.02em}
.kpi .val small{font-size:13px;color:var(--ink-2);font-weight:500;margin-left:2px}
.kpi .delta{font-family:var(--mono);font-size:10.5px;color:var(--ink-2)}
.tape{display:flex;align-items:flex-end;gap:1px;height:52px;margin-top:10px;cursor:crosshair;padding-bottom:8px;border-bottom:1px solid var(--hairline)}
.tick{flex:1;min-width:1px;border-radius:1px 1px 0 0}
.tape:hover .tick{opacity:.3}.tape .tick:hover{opacity:1}
.readout{display:flex;gap:18px;font-family:var(--mono);font-size:10.5px;color:var(--ink-3);padding-top:9px;min-height:22px}
.readout .ro-id{color:var(--cyan)}
.legend{display:flex;gap:14px;font-size:11px;color:var(--ink-2);align-items:center}
.legend i{display:inline-block;width:8px;height:8px;border-radius:2.5px;margin-right:5px}
.toolbar{display:flex;gap:10px;margin-bottom:16px;align-items:center}
.search{flex:1;display:flex;align-items:center;gap:10px;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.search:focus-within{border-color:var(--cyan-line);box-shadow:0 0 0 3px rgba(74,216,232,.08)}
.search svg{width:14px;height:14px;stroke:var(--violet);fill:none;stroke-width:2}
.search input{background:none;border:none;outline:none;color:var(--ink);font-family:var(--body);font-size:13px;width:100%}
.search input::placeholder{color:var(--ink-3)}
.search .mode{font-family:var(--mono);font-size:9.5px;color:var(--violet);background:var(--violet-soft);border:1px solid var(--violet-line);padding:2px 7px;border-radius:5px}
.corpus-layout{display:grid;grid-template-columns:184px 1fr;gap:20px}
.facet{margin-bottom:18px;font-size:12px}
.facet .fl{font-family:var(--mono);font-size:9.5px;letter-spacing:.13em;text-transform:uppercase;color:var(--ink-3);margin-bottom:8px}
.fopt{display:flex;justify-content:space-between;align-items:center;padding:5px 9px;border-radius:7px;color:var(--ink-2);cursor:pointer;user-select:none}
.fopt:hover{background:var(--surface);color:var(--ink)}.fopt.on{background:var(--cyan-soft);color:var(--cyan)}
.fopt .n{font-family:var(--mono);font-size:10px;color:var(--ink-3)}
.result-line{font-family:var(--mono);font-size:11px;color:var(--ink-3);margin-bottom:12px}.result-line b{color:var(--ink-2)}
.ep-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(214px,1fr));gap:14px}
.ep-card{background:var(--surface);border:1px solid var(--hairline);border-radius:12px;overflow:hidden;box-shadow:var(--edge-light);transition:border-color .15s,transform .15s}
.ep-card:hover{border-color:var(--line-hi);transform:translateY(-2px);box-shadow:var(--edge-light),var(--shadow-2)}
.thumb{height:120px;position:relative;overflow:hidden;background:linear-gradient(135deg,#141a24,#0d1119)}
.thumb img{width:100%;height:100%;object-fit:cover;display:block}
.thumb::after{content:'';position:absolute;inset:0;box-shadow:inset 0 -26px 24px -14px rgba(9,11,17,.7)}
.pill{position:absolute;font-family:var(--mono);font-size:9.5px;color:rgba(238,241,248,.9);background:rgba(9,11,17,.6);padding:2.5px 7px;border-radius:5px;backdrop-filter:blur(3px);z-index:2}
.pill.tl{top:8px;left:9px}.pill.br{bottom:8px;right:9px}
.q-ring{position:absolute;top:8px;right:9px;width:31px;height:31px;z-index:2}.q-ring text{font-family:var(--mono);font-size:9.5px;font-weight:500;fill:var(--ink)}
.ep-meta{padding:11px 13px 12px}
.ep-task{font-weight:600;font-size:12.5px;display:flex;align-items:center;justify-content:space-between;gap:6px}
.robot-ico{font-family:var(--mono);font-size:9.5px;color:var(--ink-3)}
.ep-id{font-family:var(--mono);font-size:10px;color:var(--ink-3);margin:2px 0 4px}
.ep-instr{font-size:11.5px;color:var(--ink-2);margin-bottom:8px;height:32px;overflow:hidden}
.chips{display:flex;flex-wrap:wrap;gap:5px}
.chip{font-family:var(--mono);font-size:9.5px;padding:2.5px 7px;border-radius:5px;background:var(--raised);color:var(--ink-2);border:1px solid var(--hairline)}
.chip.dup{color:var(--violet);border-color:var(--violet-line);background:var(--violet-soft)}
.chip.lbl{color:var(--amber);border-color:var(--amber-line);background:var(--amber-soft)}
.chip.rej{color:var(--bad);border-color:rgba(242,104,92,.25);background:var(--bad-soft)}
.btn{font-family:var(--body);font-size:12.5px;font-weight:500;padding:8px 15px;border-radius:9px;border:1px solid var(--line);background:var(--raised);color:var(--ink);cursor:pointer;box-shadow:var(--edge-light)}
.btn:hover{border-color:var(--line-hi)}.btn.primary{background:var(--amber);border-color:var(--amber);color:#1a1206;font-weight:600}
.btn.keep{border-color:rgba(65,214,147,.35);color:var(--good)}.btn.keep:hover{background:var(--good-soft)}
.btn.reject{border-color:rgba(242,104,92,.35);color:var(--bad)}.btn.reject:hover{background:var(--bad-soft)}
.dedup-head{display:flex;gap:10px;align-items:center;margin-bottom:16px;flex-wrap:wrap}
.lozenge{font-family:var(--mono);font-size:11px;color:var(--ink-2);background:var(--surface);border:1px solid var(--hairline);padding:6px 13px;border-radius:8px}
.lozenge b{color:var(--amber);font-weight:500}
.kbd-hint{font-family:var(--mono);font-size:10px;color:var(--ink-3);margin-left:auto}
.kbd-hint b{color:var(--ink-2);font-weight:400;border:1px solid var(--line);border-radius:4px;padding:0 4px;background:var(--raised)}
.pair-card{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--r-lg);padding:17px;margin-bottom:16px;box-shadow:var(--edge-light)}
.pair-card.focus{border-color:var(--amber-line);box-shadow:var(--edge-light),0 0 0 3px rgba(246,167,43,.07)}
.pair-card.decided{opacity:.55}
.pair-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:13px}
.sim-badge{font-family:var(--mono);font-size:11.5px;color:var(--violet);background:var(--violet-soft);border:1px solid var(--violet-line);padding:3.5px 11px;border-radius:7px}
.pair-grid{display:grid;grid-template-columns:1fr 44px 1fr;gap:13px;align-items:stretch}
.pair-ep{border:1px solid var(--hairline);border-radius:10px;overflow:hidden;background:var(--raised);transition:border-color .12s}
.pair-ep.pick{border-color:rgba(65,214,147,.5)}.pair-ep.drop{border-color:rgba(242,104,92,.45);opacity:.6}
.pair-ep .thumb{height:130px}.pair-ep .pm{padding:11px 13px}
.pair-ep .pm .a{font-family:var(--mono);font-size:11px;color:var(--ink-2)}
.pair-ep .pm .b{font-family:var(--mono);font-size:10px;color:var(--ink-3);margin:4px 0 7px}
.q-compare{display:flex;gap:7px;align-items:center;font-family:var(--mono);font-size:10.5px;color:var(--ink-2)}
.q-compare .qq{padding:1.5px 8px;border-radius:5px;font-weight:500}
.pair-vs{display:grid;place-items:center;font-family:var(--mono);font-size:10px;color:var(--ink-3)}
.pair-actions{display:flex;gap:9px;margin-top:13px;justify-content:flex-end}
.snap-stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
.snap-stat{background:var(--raised);border-radius:10px;padding:14px 16px;box-shadow:var(--edge-light)}
.snap-stat .a{font-family:var(--mono);font-size:9px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.11em}
.snap-stat .b{font-family:var(--disp);font-weight:600;font-size:24px;margin-top:3px}
.codebox{font-family:var(--mono);font-size:11px;color:var(--ink-2);background:var(--canvas);border:1px solid var(--hairline);border-radius:9px;padding:12px 14px;white-space:pre-wrap;line-height:1.7;margin-top:8px}
.codebox .kw{color:var(--cyan)}.codebox .vv{color:var(--amber)}
.empty{color:var(--ink-3);font-family:var(--mono);font-size:12px;padding:30px;text-align:center}
@media (max-width:1040px){#app{grid-template-columns:56px 1fr}.corpus-layout{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}.logo-name,.nav-item span,.nav-sec,.nav-foot,.nav-item .count,.ws{display:none}}
</style>
</head>
<body>
<div id="app">
<header>
  <div class="logo"><div class="logo-mark">&#9874;</div><div class="logo-name">forge <span>studio</span></div></div>
  <div class="ws"><b id="hdr-catalog">__CATALOG__</b></div>
  <span class="env">catalog &middot; static export</span>
</header>
<nav>
  <div class="nav-sec">Workspace</div>
  <div class="nav-item active" data-view="home"><svg viewBox="0 0 24 24"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg><span>Overview</span></div>
  <div class="nav-item" data-view="corpus"><svg viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1.5"/><rect x="14" y="3" width="7" height="7" rx="1.5"/><rect x="3" y="14" width="7" height="7" rx="1.5"/><rect x="14" y="14" width="7" height="7" rx="1.5"/></svg><span>Corpus</span><span class="count" id="nav-eps"></span></div>
  <div class="nav-sec">Curation</div>
  <div class="nav-item" data-view="dedup"><svg viewBox="0 0 24 24"><rect x="4" y="4" width="9" height="9" rx="2"/><rect x="11" y="11" width="9" height="9" rx="2"/></svg><span>Dedup review</span><span class="count hot" id="nav-pairs"></span></div>
  <div class="nav-item" data-view="snapshot"><svg viewBox="0 0 24 24"><path d="M12 3v12m0 0l-4-4m4 4l4-4"/><path d="M4 17v3h16v-3"/></svg><span>Snapshot</span></div>
  <div class="nav-foot"><div>embed <b id="foot-model">—</b></div><div>generated by <b>forge studio</b></div></div>
</nav>
<main>
  <section class="view active" id="v-home"></section>
  <section class="view" id="v-corpus"></section>
  <section class="view" id="v-dedup"></section>
  <section class="view" id="v-snapshot"></section>
</main>
</div>
<script>
const FORGE_DATA = /*__DATA__*/;
const THUMBS = /*__THUMBS__*/;
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const esc=s=>(s==null?'':String(s)).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const qColor=q=>q==null?'var(--ink-3)':q>=7?'var(--good)':q>=4?'var(--mid)':'var(--bad)';
const fmt=n=>(n||0).toLocaleString('en-US');
const shortId=id=>id?id.slice(0,8):'';
const D=FORGE_DATA;
$('#foot-model').textContent=(D.pairs[0]?'siglip':(D.has_embeddings?'siglip':'none'));

function thumbHTML(ep,camLabel){
  const t=ep.thumb?`<img src="${ep.thumb}" alt="">`:'';
  const dur=ep.dur?`<span class="pill br num">${ep.dur}</span>`:'';
  return `<div class="thumb">${t}${camLabel?`<span class="pill tl">${camLabel}</span>`:''}${dur}${ring(ep.score)}</div>`;
}
function ring(q){
  if(q==null)return'';
  const c=qColor(q),r=12,circ=2*Math.PI*r,off=circ*(1-q/10);
  return `<svg class="q-ring" viewBox="0 0 31 31"><circle cx="15.5" cy="15.5" r="${r}" stroke="rgba(9,11,17,.6)" stroke-width="4.5" fill="rgba(9,11,17,.55)"/><circle cx="15.5" cy="15.5" r="${r}" stroke="${c}" stroke-width="3" fill="none" stroke-dasharray="${circ}" stroke-dashoffset="${off}" stroke-linecap="round" transform="rotate(-90 15.5 15.5)"/><text x="15.5" y="19" text-anchor="middle">${q.toFixed(1)}</text></svg>`;
}

/* nav */
$$('.nav-item').forEach(n=>n.addEventListener('click',()=>{
  $$('.nav-item').forEach(x=>x.classList.remove('active'));$$('.view').forEach(x=>x.classList.remove('active'));
  n.classList.add('active');$('#v-'+n.dataset.view).classList.add('active');$('main').scrollTop=0;
}));
$('#nav-eps').textContent=fmt(D.episodes?D.episodes.length:0);
$('#nav-pairs').textContent=fmt(D.pairs.length);

/* ── OVERVIEW ── */
(()=>{
  const s=D.stats, eps=D.episodes;
  const scored=eps.filter(e=>e.score!=null).map(e=>e.score).sort((a,b)=>a-b);
  const median=scored.length?scored[Math.floor(scored.length/2)]:null;
  const approved=D.labeled.approved||0;
  const kpi=(lab,val,delta)=>`<div class="kpi"><div class="lab">${lab}</div><div class="val num">${val}</div><div class="delta">${delta||''}</div></div>`;
  let tape='';
  eps.forEach((e,i)=>{tape+=`<div class="tick" style="height:${12+Math.min(40,(e.frames||0)/40)}px;background:${qColor(e.score)}" data-i="${i}"></div>`});
  $('#v-home').innerHTML=`
   <div class="crumb"><b>${esc(D.catalog)}</b></div>
   <h1>Overview</h1>
   <div class="sub num">${fmt(s.episodes)} episodes &middot; ${fmt(s.total_frames)} frames &middot; ${s.total_hours} hours</div>
   <div class="kpis">
     ${kpi('Episodes',fmt(s.episodes),`${fmt(s.total_frames)} frames`)}
     ${kpi('Median quality',median!=null?median.toFixed(1)+'<small>/10</small>':'—','latest scorer')}
     ${kpi('Near-dup pairs',fmt(D.dedup.total_pairs),`&ge; ${D.dedup.threshold} cosine`)}
     ${kpi('Approved',fmt(approved),approved?Math.round(approved/s.episodes*100)+'% of corpus':'run forge curate')}
   </div>
   <div class="card">
     <div class="h2row"><h2>Corpus tape</h2>
       <div class="legend"><span><i style="background:var(--good)"></i>&ge;7</span><span><i style="background:var(--mid)"></i>4&ndash;7</span><span><i style="background:var(--bad)"></i>&lt;4</span><span><i style="background:var(--ink-3)"></i>unscored</span></div></div>
     <div class="tape" id="tape">${tape}</div>
     <div class="readout num" id="readout"><span>every tick is one episode &middot; height = length &middot; color = quality</span></div>
   </div>`;
  const ro=$('#readout');
  $('#tape').addEventListener('mousemove',e=>{const i=e.target.dataset&&e.target.dataset.i;if(i==null)return;const ep=eps[+i];
    ro.innerHTML=`<span class="ro-id">${shortId(ep.id)}</span><span style="color:${qColor(ep.score)}">${ep.score!=null?'q '+ep.score.toFixed(1):'unscored'}</span><span>${ep.dur}</span><span>${esc(ep.task||ep.instr||'')}</span>`;});
})();

/* ── CORPUS ── */
(()=>{
  const eps=D.episodes, pairEps=new Set();D.pairs.forEach(p=>{pairEps.add(p.a);pairEps.add(p.b)});
  const count=(k)=>{const m={};eps.forEach(e=>{const v=e[k]||'(none)';m[v]=(m[v]||0)+1});return m};
  const tasks=count('task'),robots=count('robot');
  const facet=(title,m,key)=>`<div class="facet"><div class="fl">${title}</div>`+Object.entries(m).sort((a,b)=>b[1]-a[1]).slice(0,8).map(([v,n])=>`<div class="fopt" data-facet="${key}" data-val="${esc(v)}">${esc(v)} <span class="n num">${fmt(n)}</span></div>`).join('')+`</div>`;
  const labeled=eps.filter(e=>e.label).length;
  $('#v-corpus').innerHTML=`
   <div class="crumb">${esc(D.catalog)} / <b>corpus</b></div><h1>Corpus</h1>
   <div class="sub">Browse every episode &middot; quality, labels, near-duplicates</div>
   <div class="toolbar"><div class="search"><svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/></svg>
     <input id="q" placeholder="Filter by task or instruction…"><span class="mode">text</span></div></div>
   <div class="corpus-layout"><aside>
     ${facet('Task',tasks,'task')}${facet('Robot',robots,'robot')}
     <div class="facet"><div class="fl">Label</div>
       <div class="fopt" data-facet="label" data-val="approved">approved <span class="n num">${fmt(D.labeled.approved||0)}</span></div>
       <div class="fopt" data-facet="label" data-val="rejected">rejected <span class="n num">${fmt(D.labeled.rejected||0)}</span></div></div>
   </aside><div>
     <div class="result-line"><b id="rc">${fmt(eps.length)}</b> episodes &middot; ${fmt(labeled)} labeled &middot; ${fmt(pairEps.size)} in a near-dup pair</div>
     <div class="ep-grid" id="grid"></div></div></div>`;
  let facetSel=null;
  const render=()=>{
    const q=($('#q').value||'').toLowerCase();
    const list=eps.filter(e=>{
      if(q && !((e.task||'').toLowerCase().includes(q)||(e.instr||'').toLowerCase().includes(q)))return false;
      if(facetSel && String(e[facetSel.k]||'(none)')!==facetSel.v)return false;
      return true;});
    $('#rc').textContent=fmt(list.length);
    $('#grid').innerHTML=list.slice(0,240).map(e=>`
     <div class="ep-card">${thumbHTML(e,e.robot?('robot/'+esc(e.robot)):'')}
       <div class="ep-meta"><div class="ep-task">${esc(e.task||'episode')}<span class="robot-ico">${esc(e.robot||'')}</span></div>
       <div class="ep-id">${shortId(e.id)} &middot; ${esc(e.fmt||'')}</div>
       <div class="ep-instr">${esc(e.instr||'')}</div>
       <div class="chips">${pairEps.has(e.id)?'<span class="chip dup">near-dup</span>':''}${e.label==='approved'?'<span class="chip lbl">approved</span>':e.label==='rejected'?'<span class="chip rej">rejected</span>':''}</div></div></div>`).join('')
       ||'<div class="empty">no episodes match</div>';
  };
  $('#q').addEventListener('input',render);
  $$('#v-corpus .fopt').forEach(f=>f.addEventListener('click',()=>{
    const k=f.dataset.facet,v=f.dataset.val;
    if(facetSel&&facetSel.k===k&&facetSel.v===v){facetSel=null;f.classList.remove('on');}
    else{$$('#v-corpus .fopt').forEach(x=>x.classList.remove('on'));f.classList.add('on');facetSel={k,v};}
    render();}));
  render();
})();

/* ── DEDUP ── */
(()=>{
  const pairs=D.pairs, el=$('#v-dedup');
  if(!pairs.length){el.innerHTML=`<div class="crumb">curation / <b>dedup review</b></div><h1>Dedup review</h1>
    <div class="card"><div class="empty">No near-duplicate pairs at cosine &ge; ${D.dedup.threshold}.<br>Run <b style="color:var(--cyan)">forge dedup</b> (lower <b>--threshold</b> to surface more).</div></div>`;return;}
  const decisions={};const thumbFor=id=>{const t=THUMBS[id];return t?`<img src="${t}" alt="">`:''};
  el.innerHTML=`<div class="crumb">curation / <b>dedup review</b></div><h1>Dedup review</h1>
   <div class="sub">${fmt(pairs.length)} pairs above ${D.dedup.threshold} cosine &middot; review, then apply a policy with <span class="mono" style="color:var(--cyan)">forge curate</span></div>
   <div class="dedup-head"><div class="lozenge num">decided <b id="dc">0</b> / ${pairs.length}</div>
     <button class="btn" id="export">Copy forge curate command</button>
     <div class="kbd-hint"><b>&larr;</b> keep left &nbsp;<b>&rarr;</b> keep right &nbsp;<b>x</b> reject both</div></div>
   <div id="pairs"></div>`;
  const q=(v)=>`<span class="qq num" style="background:${v>=7?'var(--good-soft)':'var(--mid-soft)'};color:${qColor(v)}">${v!=null?v.toFixed(1):'—'}</span>`;
  $('#pairs').innerHTML=pairs.map((p,i)=>{
    const pickA=(p.qa||0)>=(p.qb||0);
    return `<div class="pair-card ${i===0?'focus':''}" data-i="${i}">
     <div class="pair-head"><div style="font-family:var(--mono);font-size:11px;color:var(--ink-2)">${esc(p.task||'')} &middot; ${esc(p.robot||'')}</div>
       <span class="sim-badge num">cosine ${p.sim.toFixed(3)}</span></div>
     <div class="pair-grid">
       <div class="pair-ep ${pickA?'pick':''}" data-side="a"><div class="thumb">${thumbFor(p.a)}</div>
         <div class="pm"><div class="a">${shortId(p.a)}</div><div class="b">quality</div><div class="q-compare">${q(p.qa)}${pickA?'<span style="color:var(--good)">policy pick &check;</span>':''}</div></div></div>
       <div class="pair-vs">vs</div>
       <div class="pair-ep ${pickA?'':'pick'}" data-side="b"><div class="thumb">${thumbFor(p.b)}</div>
         <div class="pm"><div class="a">${shortId(p.b)}</div><div class="b">quality</div><div class="q-compare">${q(p.qb)}${pickA?'':'<span style="color:var(--good)">policy pick &check;</span>'}</div></div></div>
     </div>
     <div class="pair-actions"><button class="btn keep" data-act="a">Keep ${shortId(p.a)}</button>
       <button class="btn keep" data-act="b">Keep ${shortId(p.b)}</button>
       <button class="btn" data-act="both">Keep both</button>
       <button class="btn reject" data-act="reject">Reject both</button></div></div>`;}).join('');
  const decide=(i,act)=>{decisions[i]=act;const card=$(`.pair-card[data-i="${i}"]`);card.classList.add('decided');
    const eps=card.querySelectorAll('.pair-ep');eps.forEach(e=>e.classList.remove('pick','drop'));
    if(act==='a'){eps[0].classList.add('pick');eps[1].classList.add('drop');}
    if(act==='b'){eps[1].classList.add('pick');eps[0].classList.add('drop');}
    if(act==='reject'){eps.forEach(e=>e.classList.add('drop'));}
    $('#dc').textContent=Object.keys(decisions).length;};
  $('#pairs').addEventListener('click',e=>{const b=e.target.closest('[data-act]');if(!b)return;
    decide(+b.closest('.pair-card').dataset.i,b.dataset.act);});
  document.addEventListener('keydown',e=>{if(!$('#v-dedup').classList.contains('active'))return;
    const f=$('.pair-card.focus');if(!f)return;const i=+f.dataset.i;
    if(e.key==='ArrowLeft')decide(i,'a');if(e.key==='ArrowRight')decide(i,'b');if(e.key==='x')decide(i,'reject');});
  $('#export').addEventListener('click',()=>{
    const cmd=`forge curate --catalog ${D.catalog} \\\n  --dedup ${D.dedup.threshold} --dedup-policy keep-higher-quality \\\n  --label approved`;
    navigator.clipboard&&navigator.clipboard.writeText(cmd);
    $('#export').textContent='Copied ✓';setTimeout(()=>$('#export').textContent='Copy forge curate command',1400);});
})();

/* ── SNAPSHOT (Phase 4 preview) ── */
(()=>{
  const s=D.stats,L=D.labeled;
  $('#v-snapshot').innerHTML=`<div class="crumb">snapshots / <b>preview</b></div><h1>Snapshot builder</h1>
   <div class="sub">Compose a frozen, reproducible training set from curated episodes &middot; <span class="mono" style="color:var(--amber)">forge snapshot</span> lands in Phase 4</div>
   <div class="snap-stats">
     <div class="snap-stat"><div class="a">Approved</div><div class="b num" style="color:var(--good)">${fmt(L.approved||0)}</div></div>
     <div class="snap-stat"><div class="a">Rejected</div><div class="b num" style="color:var(--bad)">${fmt(L.rejected||0)}</div></div>
     <div class="snap-stat"><div class="a">Unreviewed</div><div class="b num">${fmt(s.episodes-((L.approved||0)+(L.rejected||0)+(L.held||0)))}</div></div>
   </div>
   <div class="card"><div class="h2row"><h2>Reproduce this selection today</h2><div class="meta">SQL over the catalog</div></div>
     <div class="codebox"><span class="kw">SELECT</span> e.episode_id, q.overall_score
<span class="kw">FROM</span> episodes e
<span class="kw">JOIN</span> v_curation c <span class="kw">USING</span>(episode_id)
<span class="kw">JOIN</span> v_latest_quality q <span class="kw">USING</span>(episode_id)
<span class="kw">WHERE</span> c.label = <span class="vv">'approved'</span></div>
     <div style="font-family:var(--mono);font-size:10.5px;color:var(--ink-3);margin-top:10px">Phase 4 freezes this into a snapshot manifest and materializes it to LeRobot&nbsp;v3&nbsp;/&nbsp;RLDS.</div></div>`;
})();
</script>
</body>
</html>
"""
