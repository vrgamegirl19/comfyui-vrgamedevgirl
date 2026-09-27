# Community and commercial software licensing

The root [LICENSE](../../LICENSE) contains the VRGameDevGirl Community and Commercial
License. It permits creating and monetizing media while requiring a separate paid
written agreement to commercialize covered software. The content exception includes
individuals, freelancers, and studios/businesses producing media.

Commercial licensing contact: **Jean Thompson** —
**[jeanthompson1984@gmail.com](mailto:jeanthompson1984@gmail.com)**.

## Examples

| Activity | Treatment |
| --- | --- |
| Make personal videos or learn the code | Free |
| Monetize YouTube videos, films, or music videos | Free under this code license; model/input rights still apply |
| Sell completed videos or produce a paid creative commission | Free under this code license |
| A studio uses an internal installation to make ads or client videos | Free under the content exception |
| Employees use a private production UI | Free when limited to permitted internal content production |
| A client reviews a video and asks the creator for changes | Free creative production use |
| Customers submit generation jobs through a UI, API, bot, or queue in a commercial offering | Paid software license required |
| Sell automated rendering/generation capacity while an operator submits jobs | Paid software license required |
| Fork the project, improve it, and share the fork free under the same terms | Free |
| Publish a free community fork with genuinely optional donations and no donor perks | Free |
| Charge for fork downloads, premium features, early access, or software subscriptions | Paid software license required |
| Bundle covered code into a commercial app, plugin, appliance, or service | Paid software license required |
| Offer an ad-supported generation platform using covered code | Paid software license required, even if end users pay nothing |
| Sell installation, customization, or support of the software to others | Paid software license required under this license |
| Publish a monetized tutorial about using the Builder | Free; no paid software access may be bundled |
| A company buys a commercial license | Only rights specified in its contract; no automatic copyright ownership |
| A company wants to purchase the copyright or exclusive ownership | Separately negotiated, signed rights-transfer agreement |
| Use a prior AGPL release under that release's terms | Earlier rights continue; this license cannot impose a new fee |

Examples summarize the [LICENSE](../../LICENSE); its full terms control.
Code that is independently implemented or otherwise usable without copyright
permission is not captured merely because it has similar ideas or features.

## Transition and scope

The terms start with the commit adopting them as the root LICENSE, only for code
and associated documentation Jean Thompson owns or has express authority to license
under them. They do not revoke permissions already granted for the same code or
earlier releases. Git history preserves earlier versions and their license notices.
The root LICENSE is the single current project license file; no separate historical
project license file is distributed.

ComfyUI, model weights, checkpoints, LoRAs, datasets, dependencies, and third-party
contributions retain their applicable licenses. A file being present in this
repository does not establish that Jean Thompson owns it. Neither a commercial
software license nor the media exception grants commercial rights under a model's
license or rights in music, likenesses, or other inputs.

## Maintainer review for the licensing transition

- Confirm ownership or express relicensing permission for covered contributions
  and borrowed code. Preserve third-party notices. A contribution does not
  automatically assign copyright or grant commercial relicensing authority.
- Obtain legal review of actual ComfyUI integration and incorporated copyleft code.
  A third-party exclusion alone does not resolve GPL combined-work or derivative-work
  obligations. This document is not a finding that those questions have been resolved.
- Verify package/registry compatibility with the custom license. `pyproject.toml`
  points to the root LICENSE. Describe this policy as source available, not OSI
  open source; do not label the restricted code MIT or PolyForm.
- Agree prices, payment, deployment rights, duration, support, and any ownership
  purchase in separate contracts. The community license sets no royalty percentage
  or automatic purchase entitlement. Future contributions may need separate
  contributor agreements to support commercial licensing.

## References

- [GNU licensing FAQ](https://www.gnu.org/licenses/gpl-faq.en.html): ownership,
  multiple licensing, and plugin/combined-work issues.
- [AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html): earlier license terms.
- [Open Source Definition](https://opensource.org/osd): commercial-use restrictions.

Licensing terms do not guarantee discovery of every misuse or collection of fees.
Enforceability and any remedy depend on applicable law and the relevant facts.

## Distribution metadata

The Python package metadata references the root LICENSE. The private browser-helper
package has no separate license grant; the root terms apply to covered original
code, while its dependencies retain their own licenses. Model-specific notices in
`Yue2/README.md` concern upstream materials and are not alternative licenses for
this project's original code.

The registry workflow currently publishes on manual dispatch or changes to
`pyproject.toml`, `.comfyignore`, or the publishing workflow itself. A LICENSE-only
change does not trigger it. Publish a new appropriately versioned registry package
through the existing release process when ready; prior package versions retain
the terms under which they were distributed.

## User-installed LUTs

LUT files and preview images are not bundled with the Builder. The previous
collection has been removed from the current repository.

LUT support remains available for users who supply their own licensed `.cube`
files. See [LUT installation instructions](../../LUTS/readme.txt). User-supplied
LUTs and previews remain subject to their creators' terms and are excluded from
Git and registry packages by default. This project's code license grants no
rights to those assets.
