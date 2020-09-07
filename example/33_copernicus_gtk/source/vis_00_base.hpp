#ifndef VIS_00_BASE_H
#define VIS_00_BASE_H
#include "utils.h"
;
#include "globals.h"
;
#include <iostream>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <map>
;
#include <gtkmm-3.0/gtkmm/treeview.h>
#include <gtkmm-3.0/gtkmm/liststore.h>
#include <gtkmm-3.0/gtkmm/box.h>
#include <gtkmm-3.0/gtkmm/scrolledwindow.h>
#include <gtkmm-3.0/gtkmm/window.h>
;
// header
#include <gtkmm-3.0/gtkmm/treeview.h>
#include <gtkmm-3.0/gtkmm/liststore.h>
#include <gtkmm-3.0/gtkmm/box.h>
#include <gtkmm-3.0/gtkmm/scrolledwindow.h>
#include <gtkmm-3.0/gtkmm/window.h>
 ;
class CellItem_SpacePacketHeader0  {
        public:
         CellItem_SpacePacketHeader0 ()  ;  
         ~CellItem_SpacePacketHeader0 ()  ;  
         CellItem_SpacePacketHeader0 (const CellItem_SpacePacketHeader0& src)  ;  
         CellItem_SpacePacketHeader0 (gsize offset)  ;  
        CellItem_SpacePacketHeader0& operator= (const CellItem_SpacePacketHeader0& src)  ;  
        gsize m_offset;
        gint m_packet_version_number;
        gint m_packet_type;
        gint m_secondary_header_flag;
        gint m_application_process_id_process_id;
        gint m_application_process_id_packet_category;
        gint m_sequence_flags;
        gint m_sequence_count;
        gint m_data_length;
        gint m_coarse_time;
        gint m_fine_time;
        gint m_sync_marker;
        gint m_data_take_id;
        gint m_ecc_number;
        gint m_ignore_0;
        gint m_test_mode;
        gint m_rx_channel_id;
        gint m_instrument_configuration_id;
        gint m_sub_commutated_index;
        gint m_sub_commutated_data;
        gint m_space_packet_count;
        gint m_pri_count;
        gint m_error_flag;
        gint m_ignore_1;
        gint m_baq_mode;
        gint m_baq_block_length;
        gint m_ignore_2;
        gint m_range_decimation;
        gint m_rx_gain;
        gint m_tx_ramp_rate_polarity;
        gint m_tx_ramp_rate_magnitude;
        gint m_tx_pulse_start_frequency_polarity;
        gint m_tx_pulse_start_frequency_magnitude;
        gint m_tx_pulse_length;
        gint m_ignore_3;
        gint m_rank;
        gint m_pulse_repetition_interval;
        gint m_sampling_window_start_time;
        gint m_sampling_window_length;
        gint m_sab_ssb_calibration_p;
        gint m_sab_ssb_polarisation;
        gint m_sab_ssb_temp_comp;
        gint m_sab_ssb_ignore_0;
        gint m_sab_ssb_elevation_beam_address;
        gint m_sab_ssb_ignore_1;
        gint m_sab_ssb_azimuth_beam_address;
        gint m_ses_ssb_cal_mode;
        gint m_ses_ssb_ignore_0;
        gint m_ses_ssb_tx_pulse_number;
        gint m_ses_ssb_signal_type;
        gint m_ses_ssb_ignore_1;
        gint m_ses_ssb_swap;
        gint m_ses_ssb_swath_number;
        gint m_number_of_quads;
        gint m_ignore_4;
};
class ListStore_SpacePacketHeader0 : public Gtk::Window {
        public:
         ListStore_SpacePacketHeader0 ()  ;  
         ~ListStore_SpacePacketHeader0 ()  ;  
        protected:
        void create_model ()  ;  
        void add_columns ()  ;  
        void add_items ()  ;  
        void liststore_add_item (const CellItem_SpacePacketHeader0& foo)  ;  
        Gtk::Box m_VBox;
        Gtk::ScrolledWindow m_ScrolledWindow;
        Gtk::Label m_Label;
        Gtk::TreeView m_TreeView;
        Glib::RefPtr<Gtk::ListStore> m_refListStore;
        typedef std::vector<CellItem_SpacePacketHeader0> type_vecItems;
        type_vecItems m_vecItems;
            struct ModelColumns : public Gtk::TreeModelColumnRecord {
                Gtk::TreeModelColumn<gsize> offset;
                Gtk::TreeModelColumn<gint> packet_version_number;
                Gtk::TreeModelColumn<gint> packet_type;
                Gtk::TreeModelColumn<gint> secondary_header_flag;
                Gtk::TreeModelColumn<gint> application_process_id_process_id;
                Gtk::TreeModelColumn<gint> application_process_id_packet_category;
                Gtk::TreeModelColumn<gint> sequence_flags;
                Gtk::TreeModelColumn<gint> sequence_count;
                Gtk::TreeModelColumn<gint> data_length;
                Gtk::TreeModelColumn<gint> coarse_time;
                Gtk::TreeModelColumn<gint> fine_time;
                Gtk::TreeModelColumn<gint> sync_marker;
                Gtk::TreeModelColumn<gint> data_take_id;
                Gtk::TreeModelColumn<gint> ecc_number;
                Gtk::TreeModelColumn<gint> ignore_0;
                Gtk::TreeModelColumn<gint> test_mode;
                Gtk::TreeModelColumn<gint> rx_channel_id;
                Gtk::TreeModelColumn<gint> instrument_configuration_id;
                Gtk::TreeModelColumn<gint> sub_commutated_index;
                Gtk::TreeModelColumn<gint> sub_commutated_data;
                Gtk::TreeModelColumn<gint> space_packet_count;
                Gtk::TreeModelColumn<gint> pri_count;
                Gtk::TreeModelColumn<gint> error_flag;
                Gtk::TreeModelColumn<gint> ignore_1;
                Gtk::TreeModelColumn<gint> baq_mode;
                Gtk::TreeModelColumn<gint> baq_block_length;
                Gtk::TreeModelColumn<gint> ignore_2;
                Gtk::TreeModelColumn<gint> range_decimation;
                Gtk::TreeModelColumn<gint> rx_gain;
                Gtk::TreeModelColumn<gint> tx_ramp_rate_polarity;
                Gtk::TreeModelColumn<gint> tx_ramp_rate_magnitude;
                Gtk::TreeModelColumn<gint> tx_pulse_start_frequency_polarity;
                Gtk::TreeModelColumn<gint> tx_pulse_start_frequency_magnitude;
                Gtk::TreeModelColumn<gint> tx_pulse_length;
                Gtk::TreeModelColumn<gint> ignore_3;
                Gtk::TreeModelColumn<gint> rank;
                Gtk::TreeModelColumn<gint> pulse_repetition_interval;
                Gtk::TreeModelColumn<gint> sampling_window_start_time;
                Gtk::TreeModelColumn<gint> sampling_window_length;
                Gtk::TreeModelColumn<gint> sab_ssb_calibration_p;
                Gtk::TreeModelColumn<gint> sab_ssb_polarisation;
                Gtk::TreeModelColumn<gint> sab_ssb_temp_comp;
                Gtk::TreeModelColumn<gint> sab_ssb_ignore_0;
                Gtk::TreeModelColumn<gint> sab_ssb_elevation_beam_address;
                Gtk::TreeModelColumn<gint> sab_ssb_ignore_1;
                Gtk::TreeModelColumn<gint> sab_ssb_azimuth_beam_address;
                Gtk::TreeModelColumn<gint> ses_ssb_cal_mode;
                Gtk::TreeModelColumn<gint> ses_ssb_ignore_0;
                Gtk::TreeModelColumn<gint> ses_ssb_tx_pulse_number;
                Gtk::TreeModelColumn<gint> ses_ssb_signal_type;
                Gtk::TreeModelColumn<gint> ses_ssb_ignore_1;
                Gtk::TreeModelColumn<gint> ses_ssb_swap;
                Gtk::TreeModelColumn<gint> ses_ssb_swath_number;
                Gtk::TreeModelColumn<gint> number_of_quads;
                Gtk::TreeModelColumn<gint> ignore_4;
                 ModelColumns ()    {
                        add(offset);
                        add(packet_version_number);
                        add(packet_type);
                        add(secondary_header_flag);
                        add(application_process_id_process_id);
                        add(application_process_id_packet_category);
                        add(sequence_flags);
                        add(sequence_count);
                        add(data_length);
                        add(coarse_time);
                        add(fine_time);
                        add(sync_marker);
                        add(data_take_id);
                        add(ecc_number);
                        add(ignore_0);
                        add(test_mode);
                        add(rx_channel_id);
                        add(instrument_configuration_id);
                        add(sub_commutated_index);
                        add(sub_commutated_data);
                        add(space_packet_count);
                        add(pri_count);
                        add(error_flag);
                        add(ignore_1);
                        add(baq_mode);
                        add(baq_block_length);
                        add(ignore_2);
                        add(range_decimation);
                        add(rx_gain);
                        add(tx_ramp_rate_polarity);
                        add(tx_ramp_rate_magnitude);
                        add(tx_pulse_start_frequency_polarity);
                        add(tx_pulse_start_frequency_magnitude);
                        add(tx_pulse_length);
                        add(ignore_3);
                        add(rank);
                        add(pulse_repetition_interval);
                        add(sampling_window_start_time);
                        add(sampling_window_length);
                        add(sab_ssb_calibration_p);
                        add(sab_ssb_polarisation);
                        add(sab_ssb_temp_comp);
                        add(sab_ssb_ignore_0);
                        add(sab_ssb_elevation_beam_address);
                        add(sab_ssb_ignore_1);
                        add(sab_ssb_azimuth_beam_address);
                        add(ses_ssb_cal_mode);
                        add(ses_ssb_ignore_0);
                        add(ses_ssb_tx_pulse_number);
                        add(ses_ssb_signal_type);
                        add(ses_ssb_ignore_1);
                        add(ses_ssb_swap);
                        add(ses_ssb_swath_number);
                        add(number_of_quads);
                        add(ignore_4);
};
};
    const ModelColumns m_columns;
};
void create_draw_area ()  ;  
int main (int argc, char** argv)  ;  
#endif