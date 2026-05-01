# Qualitative misclassification samples

NB-BERT correct & 8B 4-shot wrong: 238 / 4340
NB-BERT wrong & 8B 4-shot correct: 259 / 4340
Both wrong: 171 / 4340

## NB-BERT right, 8B 4-shot wrong

- **id=501663** (music, 579 words) true=positive, BERT=positive, LLM raw='negative'
  > Et velsmurt popmaskineri The Weeknd hadde en fin dag på jobben, men jobbet ikke overtid.   The Weeknd har laget noen av de største hitlåtene de siste årene.  Både han og bandet var profesjonelle til fingerspissene denne kvelden, men det var også litt av problemet: det ble akkurat litt for proft, fra begynnelse til slutt.   Bergen Live kan mer enn å lage fest med Elton John, Cecilia Brækhus og Loth...

- **id=300766** (music, 496 words) true=positive, BERT=positive, LLM raw='negative'
  > Som å se pornofilm med sladd The Weeknd har byttet inn sitt hedonistiske særpreg for prinsessa, popstjerne-status og halve bransjeriket.   Mye har skjedd siden forrige gang The Weeknd var på besøk i Oslo.  Den gang som relativt fersk superstjerne i ly av en uvanlig imponerende karriereomveltning.  Platinahoppet fra fiaskoen «Kiss Land» (2013) til «Beauty Behind the Madness» (2015) plasserte ikke b...

- **id=400704** (music, 454 words) true=negative, BERT=negative, LLM raw='positive'
  > Zara Larsson er ikke «so» god Svensken er et poptalent, men for mye av materialet når ikke opp.   Da jeg hørte gjennom dette albumet første gang, var jeg ikke spesielt imponert.  Generisk pop klar for slakting.  Lettglemt og overpolert.  Dette er ikke musikk som stikker seg nevneverdig ut fra det du hører når du blar gjennom de nye DAB-kanalene på måfå.   Men ved nøyere lytting blir man allikevel...

- **id=004769** (screen, 448 words) true=positive, BERT=positive, LLM raw='negative'
  > Flaskepost fra P Fartsfylt jakt på sadistisk seriemorder.   Flaskepost fra P er en fartsfylt spenningskrim som fungerer godt for oss som er glad i en underholdningsfokusert politijakt på sadistisk seriemorder.   Dette er den tredje filmatiseringen av Jussi Adler-Olsens bokhelt Carl Mørck, denne gangen stilsikkert regissert av norske Hans Petter Moland (Aberdeen, Kraftidioten).   Det hele er nydeli...

- **id=300464** (music, 157 words) true=positive, BERT=positive, LLM raw='negative'
  > Musikkanmeldelse:Rae Sremmurd  - «Sremmlife 2»   Crunk-vekkelsen som aldri helt når opp til klimaks.   Det er 13 år siden crunk for alvor dominerte den amerikanske rapscenen, etter at Three 6 Mafias satt startskuddet for den beinharde subsjangeren på tidlig 90-tall.  Som de fleste musikktrender, hadde også denne en relativt kort utgangsdato, så det har vært hyggelig å se den ofte undervurderte del...

## 8B 4-shot right, NB-BERT wrong

- **id=706801** (literature, 473 words) true=positive, BERT=negative, LLM raw='positive'
  > Uventa forteljing frå 2. verdskrig full av masete såpeserieeffektar BOK:  Snoflar denne romanen i sine eigne løyndomar?   Anne Swärd:  Vera.  Roman.  Omsett av Geir Pollen.  383 sider.  Gyldendal Forlag   Denne boka er så høgdramatisk.  Så overspent.  Romanen har ein “More is more”-poetikk.  Altså jo fleire bilete, historiar og dramatikk, dess betre.  Ei ung fransk gravid kvinne, Sandrine, endar o...

- **id=104007** (music, 72 words) true=positive, BERT=negative, LLM raw='positive'
  > 12 «Hold Tight» Rett over i denne.  Endelig noe fra Justins musikkmandager.  Og kveldens andre låt med det som oppleves som reelle følelser og innlevelse.  I alle fall den eneste fra «Journals» som har dukket opp fast på denne turnéen.  Her er både bandet og Bieber på.  Han løper rundt på scenen og er mer på enn han har vært hele kvelden tilsammen.  Det må være noe magisk med den gule shortsen.

- **id=301891** (music, 296 words) true=negative, BERT=positive, LLM raw='negative'
  > Kritiserer maskuline kjønnsidealer, men biter seg selv i halen Anmeldelse:  Wild Beasts - «Boy King»   ALBUM:  Wild Beasts er et glimrende drømmepopband, men det vil neppe virke sånn om «Boy King» er ditt første møte med dem.   Ikke bare er engelskmennenes femte fullengder deres svakeste så langt, den markerer også en høyresving mot et voldsommere uttrykk.  I sitt forsøk på å kritisere den overfla...

- **id=500893** (literature, 612 words) true=positive, BERT=negative, LLM raw='positive'
  > Ujevnt om fredsunderet i Colombia Fremveksten av en fredsavtale som krones med Nobels fredspris.   Den erfarne journalisten og Latin-Amerika-kjenneren Arne Halvorsen (68 år, fra Stavanger, bosatt i Rio de Janeiro i Brasil) har skrevet bok om fredsprosessen i Colombia, landet som med hjelp fra Norge nettopp har fått en fredsavtale etter 54 år med borgerkrig.   Her har det gått raskt unna, og spesie...

- **id=500188** (screen, 652 words) true=negative, BERT=positive, LLM raw='negative'
  > Portrett av idealisten som ung mann Oliver Stone supplerer med psykologiske forklaringsmodeller, men mye ved Edward Snowden fremstår like ugjennomtrengelig.   «Snowden», regissert av Oliver Stone, er basert på bøkene «The Snowden Files» og «Time of the Octopus».  Hovedpersonen selv, Edward Snowden, vil være kjent for de fleste.  En som ikke trenger så mye en introduksjon som han trenger en opphold...

## Both wrong

- **id=301687** (literature, 708 words) true=positive, BERT=negative, LLM raw='negative'
  > Ingeborg Sennesets bok om anoreksi er rystende, men bringer ikke så mye nytt Desperat kamp mot sultedøden.   «Morgen.  Våken.  Klarer ikke å stå opp.  Klarer ikke å bli liggende.  Må stå opp.  Tungt.  Jeg veier lite, men er for tung for meg selv.  Kommer meg ut av sengen.  Som en gammel kjerring.  Stygg av utseende.  Så stygg.  Styggere enn noensinne.»   Det skriver den da 24 år gamle sykepleieren...

- **id=501438** (music, 316 words) true=positive, BERT=negative, LLM raw='negative'
  > Caves svartsinn Nick Caves nye plate er full av tungsinn, men mangler melodier.   Dette albumet kommer med en ferdig skrevet kontekst:  En av Nick Caves sønner døde mens platen ble spilt inn.  Slikt appellerer til den gråtekonen i oss.  Har du anlegg for det, kan albumet lett tolkes som en lang klagesang.  Hvert sukk, hver såre vokallinje, de messende versene, den dystre poesien, den dunkle musika...

- **id=202817** (products, 1245 words) true=negative, BERT=positive, LLM raw='positive'
  > Samsung Gear 360 360-graderskamera som fanger alt – foran, bak, over og under.   Du har sikkert sett dem på Facebook – 360-gradersbildene (eller -filmene) som lar deg titte deg rundt; enten ved å bevege telefonen i lufta eller bruke pekefingeren og dra bildet i alle retninger.   Har du en VR-brille, som Samsung Gear VR, kan du også titte deg rundt med brillene på.   Man kan lage slike bilder selv...

- **id=201392** (products, 1367 words) true=positive, BERT=negative, LLM raw='negative'
  > Lydløs og syltynn hybrid fra Acer Den tredje hybrid-PC-en vi tester i år, er den første som kommer uten vifter.  Det øker lysten til å bruke den, ikke minst i nettbrettmodus.   Det siste året har vi sett mange gode hybrider mellom PC og nettbrett, ikke minst de to modellene vi testet i januar, HP Spectre 13 og Lenovo Yoga 910.   Denne gangen har vi tittet nærmere på Acers nyeste bidrag i denne kat...

- **id=104255** (music, 61 words) true=negative, BERT=positive, LLM raw='positive'
  > 9.Rune Rudberg Band:  «Run Run Away»   Tekst og melodi:  Peter Danielson, Åsa ​Karlström og Mats Larsson   Sympatisk og lettbeint danseband-country i lystig luntetrav mot avslutningens obligatoriske modulering.  Det hadde vært snillisme i praksis å sende denne til Kiev, men den får deg iallfall i godt humør.  Låta er så hyggelig og tilforlatelig at du knapt merker at den har vært der.

