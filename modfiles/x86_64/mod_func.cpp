#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;
#if defined(__cplusplus)
extern "C" {
#endif

extern void _CaDynamics_reg(void);
extern void _Ca_HVA_reg(void);
extern void _Ca_LVA_reg(void);
extern void _Ih_reg(void);
extern void _Im_reg(void);
extern void _Im_v2_reg(void);
extern void _K_P_reg(void);
extern void _K_T_reg(void);
extern void _Kd_reg(void);
extern void _Kv2like_reg(void);
extern void _Kv3_1_reg(void);
extern void _NaTa_reg(void);
extern void _NaTs_reg(void);
extern void _NaV_reg(void);
extern void _Nap_reg(void);
extern void _SK_reg(void);

void modl_reg() {
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");
    fprintf(stderr, " \"CaDynamics.mod\"");
    fprintf(stderr, " \"Ca_HVA.mod\"");
    fprintf(stderr, " \"Ca_LVA.mod\"");
    fprintf(stderr, " \"Ih.mod\"");
    fprintf(stderr, " \"Im.mod\"");
    fprintf(stderr, " \"Im_v2.mod\"");
    fprintf(stderr, " \"K_P.mod\"");
    fprintf(stderr, " \"K_T.mod\"");
    fprintf(stderr, " \"Kd.mod\"");
    fprintf(stderr, " \"Kv2like.mod\"");
    fprintf(stderr, " \"Kv3_1.mod\"");
    fprintf(stderr, " \"NaTa.mod\"");
    fprintf(stderr, " \"NaTs.mod\"");
    fprintf(stderr, " \"NaV.mod\"");
    fprintf(stderr, " \"Nap.mod\"");
    fprintf(stderr, " \"SK.mod\"");
    fprintf(stderr, "\n");
  }
  _CaDynamics_reg();
  _Ca_HVA_reg();
  _Ca_LVA_reg();
  _Ih_reg();
  _Im_reg();
  _Im_v2_reg();
  _K_P_reg();
  _K_T_reg();
  _Kd_reg();
  _Kv2like_reg();
  _Kv3_1_reg();
  _NaTa_reg();
  _NaTs_reg();
  _NaV_reg();
  _Nap_reg();
  _SK_reg();
}

#if defined(__cplusplus)
}
#endif
