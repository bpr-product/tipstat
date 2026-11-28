// HUSH product UI. Real calls only: everything goes to the workspace API in frontend/serve.py (projects, conversations,
// Tell/Ask/Do, memory timeline). A response that arrives after the user switched project or chat is ignored.
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => { const n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; };
const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } }, set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };
const S = { pid: null, cid: null, projects: [], tok: 0, facts: null, filter: 'active', q: '', kind: '', who: '', page: 'chat', highlight: null };
const rid = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random()).replace(/-/g, '');

async function call(method, path, body) {
  const r = await fetch(path, method === 'GET' ? {} : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
  return data;
}
const P = (p) => `/p/${S.pid}${p}`;
const day = (s) => (s || '').slice(0, 10);
const when = (s) => (s || '').replace('T', ' ').slice(0, 16);
const chip = (text, cls = '') => el('span', 'badge ' + cls, text);
const retryBox = (msg, fn) => { const b = el('div', 'errbox'); b.append(msg + ' '); const r = el('button', 'ghost', 'Try again'); r.onclick = fn; b.appendChild(r); return b; };

// ---- theme, navigation --------------------------------------------------------------------------------------
const wantTheme = new URLSearchParams(location.search).get('theme');
document.documentElement.dataset.theme = ['light', 'dark'].includes(wantTheme) ? wantTheme : (store.get('theme') || 'dark');
$('theme').onclick = () => { const t = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = t; store.set('theme', t); $('theme').textContent = t === 'dark' ? 'Light' : 'Dark'; };
$('theme').textContent = document.documentElement.dataset.theme === 'dark' ? 'Light' : 'Dark';
function show(page) {
  S.page = page;
  document.querySelectorAll('.page').forEach((p) => { p.hidden = p.id !== 'page-' + page; });
  document.querySelectorAll('#nav button').forEach((b) => { b.classList.toggle('on', b.dataset.page === page); b.setAttribute('aria-current', b.dataset.page === page ? 'page' : 'false'); });
  $('rail-toggle').hidden = page !== 'chat';
  if (page === 'memories') loadMemories();
  if (page === 'system') loadSystem();
}
$('nav').onclick = (e) => e.target.dataset.page && show(e.target.dataset.page);
$('rail-close').onclick = () => { $('rail').classList.remove('open'); $('rail-toggle').setAttribute('aria-expanded', 'false'); $('rail-toggle').focus(); };
$('rail-toggle').onclick = () => { const o = $('rail').classList.toggle('open'); $('rail-toggle').setAttribute('aria-expanded', String(o)); };

// ---- dialogs -------------------------------------------------------------------------------------------------------
function ask({ text, input, value = '', danger = false, ok = 'OK' }) {
  return new Promise((resolve) => {
    $('dlg-text').textContent = text; $('dlg-input').hidden = !input; $('dlg-input').value = value;
    $('dlg-ok').textContent = ok; $('dlg-ok').className = danger ? 'danger' : 'ink';
    const dlg = $('dlg'), opener = document.activeElement;
    const done = () => { dlg.removeEventListener('close', done); if (opener && opener.focus) opener.focus(); resolve(dlg.returnValue === 'ok' ? (input ? $('dlg-input').value.trim() : true) : null); };
    dlg.addEventListener('close', done); dlg.showModal(); if (input) $('dlg-input').focus();
  });
}

// ---- projects -------------------------------------------------------------------------------------------------------
async function loadProjects(prefer) {
  S.projects = (await call('GET', '/p/projects')).projects;
  const cur = S.projects.find((p) => p.id === (prefer || S.pid || store.get('pid'))) || S.projects[0];
  $('proj-name').textContent = cur.name;
  if (cur.id !== S.pid) await switchProject(cur.id); else renderProjectMenu();
}
function renderProjectMenu() {
  const m = $('proj-menu'); m.replaceChildren();
  S.projects.forEach((p) => { const b = el('button', 'menu-item' + (p.id === S.pid ? ' on' : ''), p.name); b.setAttribute('role', 'menuitemradio'); b.setAttribute('aria-checked', String(p.id === S.pid)); b.onclick = () => { closeMenu(); if (p.id !== S.pid) switchProject(p.id); }; m.appendChild(b); });
  m.appendChild(el('hr'));
  [['New project…', newProject], ['Rename this project…', renameProject], ['Delete this project…', deleteProject]].forEach(([t, fn]) => { const b = el('button', 'menu-item', t); b.setAttribute('role', 'menuitem'); b.onclick = () => { closeMenu(); fn(); }; m.appendChild(b); });
}
const closeMenu = () => { $('proj-menu').hidden = true; $('proj-btn').setAttribute('aria-expanded', 'false'); };
$('proj-btn').onclick = (e) => { e.stopPropagation(); const open = $('proj-menu').hidden; $('proj-menu').hidden = !open; $('proj-btn').setAttribute('aria-expanded', String(open)); if (open) $('proj-menu').querySelector('button').focus(); };
document.addEventListener('click', closeMenu);
async function switchProject(pid) {
  S.tok++; S.pid = pid; S.cid = null; S.facts = null; S.highlight = null; store.set('pid', pid);
  $('proj-name').textContent = S.projects.find((p) => p.id === pid).name; renderProjectMenu(); closeDrawer();
  renderTranscript([]); await loadChats();
  if (S.page === 'memories') loadMemories(); if (S.page === 'system') loadSystem();
}
async function newProject() {
  const name = await ask({ text: 'Name the new project. Its memories and conversations are completely separate.', input: true, ok: 'Create' });
  if (!name) return;
  try { const p = await call('POST', '/p/projects', { name }); await loadProjects(p.id); } catch (e) { await ask({ text: e.message }); }
}
async function renameProject() {
  const cur = S.projects.find((p) => p.id === S.pid);
  const name = await ask({ text: 'Rename this project.', input: true, value: cur.name, ok: 'Rename' });
  if (!name) return;
  try { await call('POST', `/p/projects/${S.pid}/rename`, { name }); await loadProjects(S.pid); } catch (e) { await ask({ text: e.message }); }
}
async function deleteProject() {
  const cur = S.projects.find((p) => p.id === S.pid);
  if (!(await ask({ text: `Delete "${cur.name}" with all its memories and conversations? This cannot be undone.`, danger: true, ok: 'Delete project' }))) return;
  try { await call('POST', `/p/projects/${S.pid}/delete`); S.pid = null; await loadProjects(); } catch (e) { await ask({ text: e.message }); }
}

// ---- conversations ----------------------------------------------------------------------------------------------------
async function loadChats() {
  const tok = S.tok;
  try {
    const { chats } = await call('GET', P('/chats'));
    if (tok !== S.tok) return;
    renderRail(chats);
    if (S.cid === null && chats.length) await openChat(chats[0].id);
  } catch (e) { if (tok === S.tok) $('chat-list').replaceChildren(retryBox('Could not load conversations: ' + e.message, loadChats)); }
}
function renderRail(chats) {
  const box = $('chat-list'); box.replaceChildren();
  if (!chats.length) box.appendChild(el('div', 'empty small', 'No conversations yet in this project. Start one with New chat.'));
  const today = new Date().toISOString().slice(0, 10); let last = '';
  chats.forEach((c) => {
    const group = (c.updated_ts || '').slice(0, 10) === today ? 'Today' : 'Earlier';
    if (group !== last) { box.appendChild(el('div', 'group', group)); last = group; }
    const row = el('div', 'chat-row' + (c.id === S.cid ? ' on' : '')), b = el('button', 'chat-title', c.title);
    b.title = `${c.messages} message(s) · ${when(c.updated_ts)}`; if (c.id === S.cid) b.setAttribute('aria-current', 'true');
    b.onclick = () => { $('rail').classList.remove('open'); openChat(c.id); };
    const r = el('button', 'icon', 'Rename'), d = el('button', 'icon', 'Delete');
    r.setAttribute('aria-label', `Rename conversation ${c.title}`); d.setAttribute('aria-label', `Delete conversation ${c.title}`);
    r.onclick = async () => { const t = await ask({ text: 'Rename this conversation.', input: true, value: c.title, ok: 'Rename' }); if (t) { await call('POST', P(`/chats/${c.id}/rename`), { title: t }); loadChats(); } };
    d.onclick = async () => {
      if (!(await ask({ text: 'Delete this conversation? The memories it created stay.', danger: true, ok: 'Delete' }))) return;
      await call('POST', P(`/chats/${c.id}/delete`)); if (S.cid === c.id) { S.cid = null; renderTranscript([]); }
      const tok = S.tok; const { chats: rest } = await call('GET', P('/chats')); if (tok === S.tok) { renderRail(rest); if (S.cid === null && rest.length) openChat(rest[0].id); }
    };
    row.append(b, r, d); box.appendChild(row);
  });
}
async function openChat(cid) {
  const tok = ++S.tok; S.cid = cid; closeDrawer();
  $('transcript').replaceChildren(el('div', 'empty dots', 'Loading'));
  try {
    const chat = await call('GET', P(`/chats/${cid}`));
    if (tok !== S.tok) return;
    renderTranscript(chat.messages);
    const { chats } = await call('GET', P('/chats')); if (tok === S.tok) renderRail(chats);
  } catch (e) { if (tok === S.tok) $('transcript').replaceChildren(retryBox('Could not open this conversation: ' + e.message, () => openChat(cid))); }
}
$('new-chat').onclick = async () => {
  const tok = ++S.tok;
  try { const c = await call('POST', P('/chats')); if (tok !== S.tok) return; S.cid = c.id; renderTranscript([]); const { chats } = await call('GET', P('/chats')); if (tok === S.tok) renderRail(chats); $('input').focus(); }
  catch (e) { renderTranscript([]); $('transcript').appendChild(retryBox(e.message, () => $('new-chat').click())); }
};
$('clear-chats').onclick = async () => { if (await ask({ text: 'Delete every conversation in this project? Memories stay.', danger: true, ok: 'Delete conversations' })) { await call('POST', P('/chats/clear')); S.cid = null; renderTranscript([]); loadChats(); } };

// ---- transcript ----------------------------------------------------------------------------------------------------------
const MODE_HINT = { tell: 'Tell me something to remember. To forget something, say so here.', ask: 'Ask what I know. Ask only reads memory.', do: 'Give me a job to do with the connected tools.' };
const MODE_NOTE = { tell: 'Saves what is worth keeping. Forgetting works here too.', ask: 'Answers from memory only. Never changes it.', do: 'Acts through the connected tools, checked before each change.' };
const mode = () => document.querySelector('input[name=mode]:checked').value;
document.querySelectorAll('input[name=mode]').forEach((r) => r.onchange = () => { $('input').placeholder = MODE_HINT[mode()]; $('mode-note').textContent = MODE_NOTE[mode()]; $('world-row').hidden = mode() !== 'do'; });
$('input').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('composer').requestSubmit(); } });
$('input').addEventListener('input', () => { const t = $('input'); t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight, window.innerHeight * 0.3) + 'px'; $('count').textContent = t.value.length > 3000 ? `${t.value.length}/4000` : ''; });

function renderTranscript(msgs) {
  const t = $('transcript'); t.replaceChildren();
  if (!msgs.length) {
    const w = el('section', 'welcome'); w.append(el('h1', null, 'Tell me things. Ask me later.'),
      el('p', null, 'Everything here belongs to this project only. Tell me what to remember, ask what I know, hand me a job, or ask me to forget something. Answers show the words they relied on.'));
    t.appendChild(w); return;
  }
  msgs.forEach((m) => t.appendChild(m.role === 'user' ? userBubble(m) : botBubble(m)));
  t.lastChild.scrollIntoView({ block: 'end' });
}
const MODE_NAME = { tell: 'Remember', ask: 'Ask', do: 'Do' };   // the label shown for a mode; the mode value itself is unchanged
// A small icon beside the message says how it was sent (replaces the words told / asked / job); the bubble keeps only the text.
function userBubble(m) {
  const w = el('div', 'msg user'), b = el('div', 'bubble', m.content), i = el('span', 'msg-ico');
  i.dataset.mode = m.mode; i.title = MODE_NAME[m.mode] || ''; i.setAttribute('role', 'img'); i.setAttribute('aria-label', 'Sent as ' + (MODE_NAME[m.mode] || 'a message'));
  w.append(i, b); return w;
}
function section(title, items, render) { const d = el('div', 'rc'); d.appendChild(el('strong', null, title)); items.forEach((i) => d.appendChild(el('div', 'rc-item', render ? render(i) : i))); return d; }
function usageLine(u, open) {
  if (!u) return null;
  const b = el('button', 'usage', `${u.model_calls} model call${u.model_calls === 1 ? '' : 's'} · ${u.total_tokens.toLocaleString()} tokens${u.basis === 'provider' ? '' : u.basis === 'none' ? '' : ' (est.)'} · ${(u.latency_ms / 1000).toFixed(1)} s`);
  b.setAttribute('aria-label', 'Model usage and details'); b.onclick = open; return b;
}
function botBubble(m) {
  const w = el('div', 'msg bot'), p = m.payload || {}, usage = usageLine(p.usage, () => (m.mode === 'ask' ? openAnswer(m) : m.mode === 'do' ? openJob(m) : openTell(m)));
  if (m.mode === 'tell') {
    w.classList.add('note'); const b = el('div', 'bubble'); b.appendChild(el('div', null, m.content));
    if (p.saved && p.saved.length) b.appendChild(section('Saved', p.saved));
    if (p.updated && p.updated.length) b.appendChild(section('Updated', p.updated, (u) => `${u.was}  →  ${u.now}`));
    if (p.already_known && p.already_known.length) b.appendChild(section('Already known (evidence added)', p.already_known));
    if (p.history && p.history.length) b.appendChild(section('Kept as history', p.history));
    if (p.stale && p.stale.length) b.appendChild(section('No longer valid (what they were based on changed)', p.stale));
    if (p.conflicts && p.conflicts.length) b.appendChild(section('Two current values; the later one is preferred', p.conflicts));
    if (p.verification) b.appendChild(chip(p.verification.verified ? 'removal verified' : `${p.verification.problems} problem(s) found`, p.verification.verified ? 'good' : 'bad'));
    if (p.guidance === 'ask') { const go = el('button', 'ink small', 'Ask this instead'); go.onclick = () => { const q = w.previousSibling && w.previousSibling.querySelector('.bubble'); submit('ask', q ? q.firstChild.textContent : ''); }; b.appendChild(go); }
    w.appendChild(b);
  } else if (m.mode === 'ask') {
    const ev = p.evidence || [], abstain = !ev.length || /^I (don't|do not) know|^I can only look|^I'm not sure/.test(m.content);
    if (abstain) w.classList.add('abstain');
    const b = el('div', 'bubble', m.content); b.tabIndex = 0; b.setAttribute('role', 'button'); b.setAttribute('aria-label', 'Answer. Open details.'); b.onclick = () => openAnswer(m); b.onkeydown = (e) => { if (e.key === 'Enter') openAnswer(m); };
    const readOnly = (p.trace || []).some((t) => t.step === 'read_only');
    w.append(b, el('div', 'meta', readOnly ? 'Ask only reads memory. Use Tell to store, correct or forget something.' : ev.length ? `${ev.length} source(s). Open the answer to inspect them.` : 'No supporting memory. Open for details.'));
    if (p.follow_up_of) w.appendChild(el('div', 'meta', `Understood as a follow-up to: “${p.follow_up_of}”`));
  } else {
    const b = el('div', 'bubble', m.content); b.tabIndex = 0; b.setAttribute('role', 'button'); b.setAttribute('aria-label', 'Reply. Open details.'); b.onclick = () => openJob(m); b.onkeydown = (e) => { if (e.key === 'Enter') openJob(m); };
    const tr = p.trace || [], n = tr.filter((t) => t.step === 'tool').length, blocked = tr.filter((t) => t.step === 'blocked').length;
    w.append(b, el('div', 'meta', `${n} tool call(s)${blocked ? `, ${blocked} blocked by a safety check` : ''}. Open the reply to inspect them.`));
  }
  if (usage) w.appendChild(usage);
  return w;
}
async function submit(m, text, node, requestId) {
  const tok = S.tok, id = requestId || rid();
  if (S.cid === null) { const c = await call('POST', P('/chats')); if (tok !== S.tok) return; S.cid = c.id; }
  const box = $('transcript'); if (box.querySelector('.welcome')) box.replaceChildren();
  const pending = node || el('div', 'msg bot');
  if (!node) { const u = userBubble({ content: text, mode: m }); box.appendChild(u); box.appendChild(pending); pending.userNode = u; }
  pending.className = 'msg bot'; pending.setAttribute('aria-busy', 'true');
  pending.replaceChildren(el('div', 'bubble dots', m === 'tell' ? 'Saving' : m === 'ask' ? 'Looking' : 'Working')); pending.scrollIntoView({ block: 'end' });
  try {
    const out = await call('POST', P(`/chats/${S.cid}/send`), { mode: m, text, world: $('world').value.trim(), request_id: id });
    if (tok !== S.tok) return;                                  // the user moved on: do not draw into another project or chat
    const a = out.assistant.payload || {};
    if (a.forgotten || String(a.extraction || '').startsWith('forget')) { await openChat(S.cid); S.facts = null; return; }   // saved chat content may have changed: reload it
    pending.userNode.replaceWith(userBubble(out.user)); pending.replaceWith(botBubble(out.assistant));
    const { chats } = await call('GET', P('/chats')); if (tok === S.tok) renderRail(chats);
    S.facts = null;
  } catch (e) {
    if (tok !== S.tok) return;
    pending.className = 'msg bot error'; pending.removeAttribute('aria-busy');
    const retry = el('button', 'ghost', 'Retry'); retry.onclick = () => submit(m, text, pending, id);   // same request id: it can never be applied twice
    const meta = el('div', 'meta'); meta.append('Nothing was applied twice and nothing was saved from this attempt. ', retry);
    pending.replaceChildren(el('div', 'bubble', e.message), meta);
  }
}
$('composer').onsubmit = async (e) => {
  e.preventDefault(); const text = $('input').value.trim(), m = mode(); if (!text) return;
  $('input').value = ''; $('input').style.height = 'auto'; $('count').textContent = ''; $('send').disabled = true; await submit(m, text); $('send').disabled = false; $('input').focus();
};

// ---- drawers: answer details, job details, memory inspector ---------------------------------------------------------------
let drawerOpener = null;
function drawer(title, build) {
  drawerOpener = document.activeElement; $('drawer-title').textContent = title;
  const b = $('drawer-body'); b.replaceChildren(); build(b); b.scrollTop = 0;
  $('drawer').hidden = $('scrim').hidden = false; $('close').focus();
}
function closeDrawer() { if ($('drawer').hidden) return; $('drawer').hidden = $('scrim').hidden = true; if (drawerOpener && document.contains(drawerOpener) && drawerOpener.focus) drawerOpener.focus(); }
$('close').onclick = $('scrim').onclick = closeDrawer;
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') { closeDrawer(); closeMenu(); }
  if (e.key === 'Tab' && !$('drawer').hidden) {                // keep keyboard focus inside the open drawer
    const f = [...$('drawer').querySelectorAll('button, [tabindex="0"], summary, a')].filter((x) => !x.disabled);
    if (!f.length) return; const first = f[0], last = f[f.length - 1];
    if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); } else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
  }
});
const h = (b, t) => b.appendChild(el('h4', null, t));
function raw(b, obj) { const d = el('details', 'raw'); d.appendChild(el('summary', null, 'Raw diagnostic data')); d.appendChild(el('pre', null, JSON.stringify(obj, null, 2))); b.appendChild(d); }
function usageBlock(b, u) {
  if (!u) return; h(b, 'Model usage'); const t = el('dl', 'usage-grid');
  const row = (k, v) => t.append(el('dt', null, k), el('dd', null, v));
  row('Model calls', String(u.model_calls) + (u.failed_calls ? ` (${u.failed_calls} failed)` : ''));
  row('Input tokens', u.input_tokens.toLocaleString()); row('Output tokens', u.output_tokens.toLocaleString()); row('Total tokens', u.total_tokens.toLocaleString());
  row('Counted as', u.basis === 'provider' ? 'Reported by the provider' : u.basis === 'mixed' ? 'Mixed: provider-reported for some calls, estimated for others' : u.basis === 'none' ? 'No model call was made' : 'Estimated (about 4 characters per token); the provider reported no usage');
  row('Latency', `${(u.latency_ms / 1000).toFixed(2)} s in total, ${(u.model_ms / 1000).toFixed(2)} s waiting for the model`);
  b.appendChild(t);
}
function statusChip(f) { return f.status === 'active' ? chip('current', 'good') : chip(f.status === 'stale' ? `no longer valid ${day(f.replaced)}` : `earlier version, replaced ${day(f.replaced)}`, 'history'); }
function evidenceCard(c) {
  const d = el('div', 'evcard'); d.appendChild(el('div', 'evlabel', c.label));
  d.appendChild(el('blockquote', 'evquote', `“${c.quote}”`));
  const chips = el('div', 'chips-row');
  if (c.kind !== 'source') chips.append(chip(c.kind.toLowerCase()), chip(c.attribution === 'other' ? 'said by someone else' : 'about you', c.attribution === 'other' ? 'other' : ''), statusChip(c));
  else chips.append(chip('source message'));
  chips.append(chip(`${c.source_kind === 'event' ? 'app event' : 'chat'}${c.source_origin ? ' · ' + c.source_origin : ''} · message ${c.message_id}`));
  d.appendChild(chips);
  if (c.fact_id) { const open = el('button', 'ghost small', `Open memory #${c.fact_id}`); open.onclick = () => openMemory(c.fact_id); d.appendChild(open); }
  return d;
}
function flowStep(n, title, body, state = 'on') { const li = el('li', 'step ' + state); li.append(el('span', 'dot', String(n)), el('div', 'stepbody')); li.lastChild.appendChild(el('strong', null, title)); li.lastChild.append(...[].concat(body)); return li; }
function refList(items, big) { const d = el('details', 'raw'); d.appendChild(el('summary', null, `${items.length} item(s)`)); items.forEach((c) => { const r = el('div', 'cand'); r.append(chip(c.ref, 'mono'), el('span', null, c.label || ''), ...(c.via ? [chip(c.via)] : []), ...(c.status && c.status !== 'active' && c.status !== 'source' ? [chip(c.status, 'history')] : [])); d.appendChild(r); }); return d; }
function openAnswer(m) {
  const p = m.payload || {}, f = p.flow || {}, ev = p.cited || [];
  drawer('Answer details', (b) => {
    h(b, ev.length ? 'Evidence used' : 'No evidence cited');
    if (!ev.length) b.appendChild(el('div', 'muted', 'Nothing stored supports a firm answer, so none is shown.'));
    ev.forEach((c) => b.appendChild(evidenceCard(c)));
    if (p.flow) {
      h(b, 'How it was found'); const ol = el('ol', 'steps'), na = el('span', 'muted', 'not available');
      const ex = f.expansion || {};
      ol.appendChild(flowStep(1, 'Keyword expansion', ex.terms && ex.terms.length ? el('div', 'chips-row', null) : na, ex.terms && ex.terms.length ? 'on' : 'off'));
      if (ex.terms && ex.terms.length) ol.lastChild.querySelector('.chips-row').append(...ex.terms.map((t) => chip(t)));
      else if (ex.error) ol.lastChild.append(el('span', 'muted', ` model call failed (${ex.error}); keyword search only`));
      ol.appendChild(flowStep(2, 'Lexical search', f.lexical == null ? na : el('span', null, ` ${f.lexical} hit(s)`), f.lexical == null ? 'off' : 'on'));
      ol.appendChild(flowStep(3, 'Semantic search', f.semantic_used ? el('span', null, ` ${f.semantic_hits} hit(s)`) : el('span', 'muted', ' off: no embedding model is ready'), f.semantic_used ? 'on' : 'off'));
      ol.appendChild(flowStep(4, 'Rank fusion (RRF)', f.rrf_applied ? el('span', null, ` ${f.fused} ranked from both methods`) : el('span', 'muted', ' not applied: only one method returned results'), f.rrf_applied ? 'on' : 'off'));
      ol.appendChild(flowStep(5, 'Candidates considered', [el('span', null, ` ${(f.candidates || []).length}`), refList(f.candidates || [])]));
      ol.appendChild(flowStep(6, 'Supplied to the model', [el('span', null, ` ${(f.supplied || []).length} item(s)${f.supplied_tokens ? `, about ${f.supplied_tokens} tokens` : ''}`), refList(f.supplied || [])]));
      ol.appendChild(flowStep(7, 'Cited in the answer', el('span', null, ` ${ev.length}`)));
      (f.extra || []).forEach((x) => ol.appendChild(flowStep('+', x.step === 'parts' ? 'Each part of the question was also searched' : x.step === 'agenda' ? 'Due-soon and open items were added' : x.step === 'abstain' ? 'Answer withheld' : 'Read-only', el('span', 'muted', x.n != null ? ` ${x.n}` : ` ${x.reason || ''}`))));
      b.appendChild(ol);
    }
    usageBlock(b, p.usage); raw(b, { evidence: p.evidence, trace: p.trace });
  });
}
function openTell(m) { const p = m.payload || {}; drawer('Details', (b) => { h(b, 'Result'); b.appendChild(el('div', null, m.content)); usageBlock(b, p.usage); raw(b, p); }); }
function openJob(m) {
  const p = m.payload || {}, t = p.trace || [];
  drawer('Job details', (b) => {
    h(b, 'Steps taken'); t.filter((x) => x.step === 'tool').forEach((x) => { const d = el('div', 'evcard'); d.append(el('div', 'evlabel', x.name), el('div', 'step-mono', x.args), el('div', 'muted step-mono', '→ ' + x.result)); b.appendChild(d); });
    if (!t.some((x) => x.step === 'tool')) b.appendChild(el('div', 'muted', 'No tools were called.'));
    const blocked = t.filter((x) => x.step === 'blocked');
    if (blocked.length) { h(b, 'Blocked by a safety check'); blocked.forEach((x) => { const d = el('div', 'evcard bad'); d.append(el('div', 'evlabel', x.name), el('div', null, x.reason)); b.appendChild(d); }); }
    usageBlock(b, p.usage); raw(b, t);
  });
}
async function openMemory(fid) {
  const tok = S.tok;
  try {
    const m = await call('GET', P(`/memory/${fid}`)); if (tok !== S.tok) return;
    drawer(`Memory #${m.fact.id}`, (b) => {
      const f = m.fact; b.appendChild(el('div', 'evlabel', `${f.subject} · ${f.relation}: ${f.value}`));
      b.appendChild(el('blockquote', 'evquote', `“${f.quote}”`));
      const chips = el('div', 'chips-row'); chips.append(chip(f.kind.toLowerCase()), chip(f.attribution === 'other' ? 'said by someone else' : 'about you', f.attribution === 'other' ? 'other' : ''),
        statusChip({ status: f.status, replaced: f.ended_ts }), chip(`noted ${day(f.created_ts)}`)); if (f.when_ts || f.when_date) chips.append(chip('date ' + when(f.when_ts || f.when_date)));
      if (f.sensitive) chips.append(chip('private', 'bad')); b.appendChild(chips);
      h(b, 'Evidence'); m.evidence.forEach((e) => { const d = el('div', 'evcard'); d.append(el('blockquote', 'evquote', `“${e.quote}”`), el('div', 'muted', `${e.source_kind === 'event' ? 'app event' : 'chat'}${e.origin ? ' · ' + e.origin : ''} · message ${e.message_id} · ${when(e.ts)}`));
        const det = el('details', 'raw'); det.appendChild(el('summary', null, 'Source text')); det.appendChild(el('pre', null, e.source_text)); d.appendChild(det); b.appendChild(d); });
      if (!m.evidence.length) b.appendChild(el('div', 'muted', 'No evidence rows.'));
      const link = (r) => { const x = el('button', 'ghost small', `#${r.id} ${r.label}`); x.onclick = () => openMemory(r.id); return x; };
      if (m.earlier_versions.length || m.replaced_by.length) { h(b, 'Version timeline'); m.replaced_by.forEach((id) => b.appendChild(el('div', 'ver', `Replaced by memory #${id}`)));
        b.appendChild(el('div', 'ver', `This version: noted ${day(f.created_ts)}${f.ended_ts ? `, ended ${day(f.ended_ts)}` : ' (current)'}`));
        m.earlier_versions.forEach((v) => { const d = el('div', 'ver'); d.append(link(v), el('span', 'muted', ` noted ${day(v.noted)}${v.ended ? `, replaced ${day(v.ended)}` : ''}`)); b.appendChild(d); }); }
      if (m.derived_from.length) { h(b, 'Derived from'); m.derived_from.forEach((r) => b.appendChild(link(r))); }
      if (m.derived_facts.length) { h(b, 'Facts derived from this'); m.derived_facts.forEach((r) => b.appendChild(link(r))); }
      const show_ = el('button', 'ghost small', 'Show in Memories'); show_.onclick = () => { S.highlight = f.id; S.filter = 'all'; closeDrawer(); show('memories'); }; b.appendChild(show_);
    });
  } catch (e) { drawer('Memory', (b) => b.appendChild(retryBox('Could not open this memory: ' + e.message, () => openMemory(fid)))); }
}

// ---- memories: current, timeline, all versions, search and filters --------------------------------------------------------------
$('filters').onclick = (e) => { if (!e.target.dataset.f) return; S.filter = e.target.dataset.f; syncFilters(); renderMemories(); };
$('mem-q').oninput = () => { S.q = $('mem-q').value.trim().toLowerCase(); renderMemories(); };
$('mem-kind').onchange = () => { S.kind = $('mem-kind').value; renderMemories(); };
$('mem-who').onchange = () => { S.who = $('mem-who').value; renderMemories(); };
$('refresh').onclick = loadMemories;
function syncFilters() { document.querySelectorAll('#filters button').forEach((b) => { const on = b.dataset.f === S.filter; b.classList.toggle('on', on); b.setAttribute('aria-pressed', String(on)); }); }
async function loadMemories() {
  const tok = S.tok; syncFilters(); $('facts').replaceChildren(el('div', 'empty dots', 'Loading'));
  try { const m = await call('GET', P('/memory')); if (tok !== S.tok) return; S.facts = m; renderMemories(); }
  catch (e) { if (tok === S.tok) $('facts').replaceChildren(retryBox("Could not read this project's memory: " + e.message, loadMemories)); }
}
function factCard(f, by) {
  const card = el('div', 'card' + (f.status === 'active' ? '' : ' old') + (S.highlight === f.id ? ' flash' : '')), head = el('div', 'head');
  card.id = 'fact-' + f.id;
  const title = el('button', 'val link', `${f.subject} · ${f.relation}: ${f.value}`); title.onclick = () => openMemory(f.id);
  head.append(title, chip(f.kind.toLowerCase()), chip(f.attribution === 'other' ? 'said by someone else' : 'about you', f.attribution === 'other' ? 'other' : ''));
  if (f.when_ts || f.when_date) head.append(chip(when(f.when_ts || f.when_date)));
  if (f.sensitive) head.append(chip('private', 'bad'));
  if (f.status === 'superseded') head.append(chip('replaced ' + day(f.ended_ts), 'history'));
  if (f.status === 'stale') head.append(chip('no longer valid ' + day(f.ended_ts), 'history'));
  if (f.conflict_with && by[f.conflict_with] && by[f.conflict_with].status === 'active') head.append(chip(`conflicts with #${f.conflict_with}`, 'bad'));
  if (f.derived_from.length) head.append(chip('derived from ' + f.derived_from.map((i) => '#' + i).join(', ')));
  card.append(head, el('div', 'quote', `“${f.quote}”`), el('div', 'muted', `#${f.id} · from message ${f.first_message_id} · noted ${day(f.created_ts)} · ${f.evidence_count} source(s)`));
  return card;
}
function renderMemories() {
  const m = S.facts; if (!m) return;
  const box = $('facts'); box.replaceChildren(); const by = Object.fromEntries(m.facts.map((f) => [f.id, f]));
  $('counts').textContent = `${m.counts.active} current · ${m.counts.history} earlier · ${m.counts.sources} source(s)`;
  let rows = m.facts;
  if (S.filter === 'active') rows = rows.filter((f) => f.status === 'active');
  if (S.filter === 'timeline') rows = rows.filter((f) => f.is_head).sort((a, b) => (b.created_ts || '').localeCompare(a.created_ts || ''));
  if (S.kind) rows = rows.filter((f) => f.kind === S.kind);
  if (S.who) rows = rows.filter((f) => f.attribution === S.who);
  if (S.q) rows = rows.filter((f) => `${f.subject} ${f.relation} ${f.value} ${f.quote}`.toLowerCase().includes(S.q));
  if (!rows.length) box.appendChild(el('div', 'empty', m.counts.sources ? 'Nothing matches this view and these filters.' : 'Nothing is stored in this project yet. Tell me something on the Chat page.'));
  rows.forEach((f) => {
    const card = factCard(f, by);
    if (S.filter === 'timeline' && f.versions.length) {
      const hist = el('div', 'versions'); hist.appendChild(el('h4', null, 'Earlier versions'));
      f.versions.forEach((id) => { const v = by[id]; hist.appendChild(el('div', 'ver', `${v.value}  ·  noted ${day(v.created_ts)}${v.ended_ts ? `, replaced ${day(v.ended_ts)}` : ''}  ·  “${v.quote}”`)); });
      card.appendChild(hist);
    }
    box.appendChild(card);
  });
  if (S.highlight) { const t = $('fact-' + S.highlight); if (t) t.scrollIntoView({ block: 'center' }); S.highlight = null; }
  const fb = $('forgets'); fb.replaceChildren();
  if (!m.forgets.length) fb.appendChild(el('div', 'empty', 'Nothing has been forgotten in this project.'));
  m.forgets.forEach((r) => { const d = el('div', 'card'), hd = el('div', 'head');
    hd.append(el('span', null, `${r.facts_removed} fact(s) and ${r.sources_removed} source(s) erased`), chip(r.verified ? 'verified' : `${r.problems} problem(s)`, r.verified ? 'good' : 'bad'), el('span', 'muted', day(r.ts)));
    d.appendChild(hd); fb.appendChild(d); });
}

// ---- system ---------------------------------------------------------------------------------------------------------------------
async function loadSystem() {
  const dl = $('sys'), idx = $('index'); dl.replaceChildren(); idx.replaceChildren(); const row = (k, v) => dl.append(el('dt', null, k), el('dd', null, v));
  try {
    const s = await call('GET', '/p/system');
    row('Projects', String(s.projects)); row('Product storage', s.root + '  (separate from the evaluation database)');
    row('Model', s.model + (s.model_configured ? '' : '  (no API key configured: set OPENAI_API_KEY in .env)'));
    const sem = el('span'); sem.append(chip(s.semantic.ready ? 'ready' : 'not ready', s.semantic.ready ? 'good' : 'bad'), ` ${s.semantic.model}${s.semantic.ready ? '' : ': ' + s.semantic.reason + '. Keyword search is used instead.'}`);
    dl.append(el('dt', null, 'Semantic search'), (() => { const d = el('dd'); d.appendChild(sem); return d; })());
    const tbl = el('table', 'tbl'); tbl.appendChild(el('caption', null, 'Search index per project'));
    const head = el('tr'); ['Project', 'Facts indexed', 'Messages indexed', ''].forEach((t) => head.appendChild(el('th', null, t))); tbl.appendChild(head);
    s.index.forEach((p) => {
      const r = el('tr'), missing = (p.facts - p.facts_indexed) + (p.sources - p.sources_indexed);
      r.append(el('td', null, p.name), el('td', null, `${p.facts_indexed} of ${p.facts}`), el('td', null, `${p.sources_indexed} of ${p.sources}`));
      const c = el('td'); if (missing && s.semantic.ready) { const b = el('button', 'ghost small', `Index ${missing} missing`); b.onclick = async () => { b.disabled = true; try { await call('POST', `/p/${p.id}/reindex`); } catch (e) { await ask({ text: e.message }); } loadSystem(); }; c.appendChild(b); } else if (missing) c.appendChild(el('span', 'muted', 'waiting for the model')); else c.appendChild(chip('complete', 'good'));
      r.appendChild(c); tbl.appendChild(r);
    });
    idx.appendChild(tbl);
  } catch (e) { dl.replaceChildren(); idx.appendChild(retryBox('Could not read the system status: ' + e.message, loadSystem)); }
}
$('reset').onclick = async () => { if (await ask({ text: 'Delete every project, memory and conversation in this product? This cannot be undone.', danger: true, ok: 'Delete all' })) { try { await call('POST', '/p/reset-all'); S.pid = null; await loadProjects(); loadSystem(); } catch (e) { await ask({ text: 'Failed: ' + e.message }); } } };

loadProjects().then(() => show('chat')).catch((e) => { $('transcript').replaceChildren(retryBox('The workspace is not reachable: ' + e.message, () => location.reload())); });
