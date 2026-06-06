export const TEAM_META = {
  "Mexico":                 ["🇲🇽", "MEX"],
  "South Korea":            ["🇰🇷", "KOR"],
  "South Africa":           ["🇿🇦", "RSA"],
  "Czech Republic":         ["🇨🇿", "CZE"],
  "Canada":                 ["🇨🇦", "CAN"],
  "Switzerland":            ["🇨🇭", "SUI"],
  "Qatar":                  ["🇶🇦", "QAT"],
  "Bosnia & H.":            ["🇧🇦", "BIH"],
  "Bosnia and Herzegovina": ["🇧🇦", "BIH"],
  "Brazil":                 ["🇧🇷", "BRA"],
  "Morocco":                ["🇲🇦", "MAR"],
  "Scotland":               ["🏴󠁧󠁢󠁳󠁣󠁴󠁿", "SCO"],
  "Haiti":                  ["🇭🇹", "HAI"],
  "United States":          ["🇺🇸", "USA"],
  "Paraguay":               ["🇵🇾", "PAR"],
  "Australia":              ["🇦🇺", "AUS"],
  "Turkey":                 ["🇹🇷", "TUR"],
  "Germany":                ["🇩🇪", "GER"],
  "Ecuador":                ["🇪🇨", "ECU"],
  "Ivory Coast":            ["🇨🇮", "CIV"],
  "Curaçao":                ["🇨🇼", "CUW"],
  "Netherlands":            ["🇳🇱", "NED"],
  "Japan":                  ["🇯🇵", "JPN"],
  "Tunisia":                ["🇹🇳", "TUN"],
  "Sweden":                 ["🇸🇪", "SWE"],
  "Belgium":                ["🇧🇪", "BEL"],
  "Iran":                   ["🇮🇷", "IRN"],
  "Egypt":                  ["🇪🇬", "EGY"],
  "New Zealand":            ["🇳🇿", "NZL"],
  "Spain":                  ["🇪🇸", "ESP"],
  "Uruguay":                ["🇺🇾", "URU"],
  "Saudi Arabia":           ["🇸🇦", "KSA"],
  "Cape Verde":             ["🇨🇻", "CPV"],
  "France":                 ["🇫🇷", "FRA"],
  "Senegal":                ["🇸🇳", "SEN"],
  "Norway":                 ["🇳🇴", "NOR"],
  "Iraq":                   ["🇮🇶", "IRQ"],
  "Argentina":              ["🇦🇷", "ARG"],
  "Austria":                ["🇦🇹", "AUT"],
  "Algeria":                ["🇩🇿", "ALG"],
  "Jordan":                 ["🇯🇴", "JOR"],
  "Portugal":               ["🇵🇹", "POR"],
  "Colombia":               ["🇨🇴", "COL"],
  "Uzbekistan":             ["🇺🇿", "UZB"],
  "DR Congo":               ["🇨🇩", "COD"],
  "England":                ["🏴󠁧󠁢󠁥󠁮󠁧󠁿", "ENG"],
  "Croatia":                ["🇭🇷", "CRO"],
  "Panama":                 ["🇵🇦", "PAN"],
  "Ghana":                  ["🇬🇭", "GHA"],
}

export const NAME_SHORT = {
  "Bosnia and Herzegovina": "Bosnia & H.",
}

export function tmeta(t) {
  return TEAM_META[t] || ["🌍", t.slice(0, 3).toUpperCase()]
}

export function tshort(t) {
  return NAME_SHORT[t] || t
}

export function tdisplay(t) {
  const [flag] = tmeta(t)
  return `${flag} ${t}`
}
