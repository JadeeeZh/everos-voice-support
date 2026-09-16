# Starter-pack call data

Twelve real support calls from the measured experiment's learning
split, rendered as POST-ready `/api/v2/memory/add` payloads. Failed
or escalated calls end with the tier-2 specialist's factual
close-out note — the correction signal a real call center records.

| file | family | injected faults | outcome | correction note |
|---|---|---|---|---|
| `calls/call-001.json` | mobile_data_issue | `airplane_mode_on|user_abroad_roaming_enabled_off` | resolved | — |
| `calls/call-002.json` | mobile_data_issue | `airplane_mode_on|data_mode_off` | resolved | — |
| `calls/call-003.json` | mobile_data_issue | `airplane_mode_on|bad_network_preference` | resolved | — |
| `calls/call-004.json` | mobile_data_issue | `bad_network_preference|user_abroad_roaming_enabled_off` | transferred | yes |
| `calls/call-005.json` | mobile_data_issue | `data_usage_exceeded|user_abroad_roaming_enabled_off` | transferred | yes |
| `calls/call-006.json` | mobile_data_issue | `data_mode_off|data_usage_exceeded` | resolved | — |
| `calls/call-007.json` | mobile_data_issue | `bad_vpn|data_saver_mode_on|user_abroad_roaming_disabled_on` | resolved | — |
| `calls/call-008.json` | mobile_data_issue | `data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off` | failed | yes |
| `calls/call-009.json` | mms_issue | `break_app_storage_permission|data_usage_exceeded` | resolved | — |
| `calls/call-010.json` | mms_issue | `bad_network_preference|user_abroad_roaming_disabled_on` | resolved | — |
| `calls/call-011.json` | mms_issue | `bad_network_preference|user_abroad_roaming_disabled_off` | transferred | yes |
| `calls/call-012.json` | mms_issue | `break_apn_mms_setting|data_mode_off|user_abroad_roaming_disabled_on` | resolved | — |
