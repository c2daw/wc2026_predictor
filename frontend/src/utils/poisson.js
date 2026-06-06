function poissonPmf(k, lambda) {
  if (lambda <= 0) return k === 0 ? 1 : 0
  let logP = k * Math.log(Math.max(lambda, 1e-10)) - lambda
  for (let i = 1; i <= k; i++) logP -= Math.log(i)
  return Math.exp(logP)
}

function rhoCorrection(x, y, lam, mu, rho) {
  if (x === 0 && y === 0) return 1 - lam * mu * rho
  if (x === 0 && y === 1) return 1 + lam * rho
  if (x === 1 && y === 0) return 1 + mu * rho
  if (x === 1 && y === 1) return 1 - rho
  return 1.0
}

export function predictMatch(teamA, teamB, modelData, maxGoals = 8) {
  const { valid_teams, params, rho } = modelData
  const n = valid_teams.length
  const attack = params.slice(0, n)
  const defense = params.slice(n, 2 * n)

  const ai = valid_teams.indexOf(teamA)
  const bi = valid_teams.indexOf(teamB)
  if (ai === -1 || bi === -1) return null

  const lam = Math.exp(attack[ai] - defense[bi])
  const mu = Math.exp(attack[bi] - defense[ai])

  const matrix = []
  let total = 0
  for (let x = 0; x <= maxGoals; x++) {
    const row = []
    for (let y = 0; y <= maxGoals; y++) {
      const rc = rhoCorrection(x, y, lam, mu, rho)
      const p = Math.max(rc * poissonPmf(x, lam) * poissonPmf(y, mu), 0)
      row.push(p)
      total += p
    }
    matrix.push(row)
  }

  const norm = matrix.map(row => row.map(v => v / total))

  let winA = 0, draw = 0, winB = 0
  for (let x = 0; x <= maxGoals; x++) {
    for (let y = 0; y <= maxGoals; y++) {
      if (x > y) winA += norm[x][y]
      else if (x === y) draw += norm[x][y]
      else winB += norm[x][y]
    }
  }

  return { lam, mu, winA, draw, winB, matrix: norm }
}

export function collapseMatrix(matrix, maxGoals = 8) {
  const MAX_G = 4
  const nc = MAX_G + 2
  const sm = Array.from({ length: nc }, () => Array(nc).fill(0))
  for (let r = 0; r < nc; r++) {
    for (let c = 0; c < nc; c++) {
      const rMin = r <= MAX_G ? r : MAX_G + 1
      const rMax = r <= MAX_G ? r : maxGoals
      const cMin = c <= MAX_G ? c : MAX_G + 1
      const cMax = c <= MAX_G ? c : maxGoals
      let sum = 0
      for (let ri = rMin; ri <= rMax; ri++) {
        for (let ci = cMin; ci <= cMax; ci++) {
          sum += matrix[ri][ci]
        }
      }
      sm[r][c] = sum * 100
    }
  }
  return sm
}

export function topScorelines(matrix, topN = 8, maxGoals = 8) {
  const rows = []
  for (let x = 0; x <= maxGoals; x++) {
    for (let y = 0; y <= maxGoals; y++) {
      rows.push({ x, y, p: matrix[x][y] })
    }
  }
  return rows.sort((a, b) => b.p - a.p).slice(0, topN)
}
