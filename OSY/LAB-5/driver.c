#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kdev_t.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#define mem_size 1024 //Memory Size

dev_t dev = 0;
static struct class* dev_class;
static struct cdev ex_cdev;
uint8_t* kernel_buffer;

static int __init ex_driver_init(void);
static void __exit ex_driver_exit(void);
static int ex_open(struct inode* inode, struct file* file);
static int ex_release(struct inode* inode, struct file* file);
static ssize_t ex_read(struct file* filp, char __user* buf, size_t len,
		loff_t* off);
static ssize_t ex_write(struct file* filp, const char* buf, size_t len,
						loff_t* off);

static struct file_operations fops = {
		.owner = THIS_MODULE,
		.read = ex_read,
		.write = ex_write,
		.open = ex_open,
		.release = ex_release,
};

static int ex_open(struct inode* inode, struct file* file)
{
	pr_info("Device File Opened...!!!\n");
	return 0;
}

static int ex_release(struct inode* inode, struct file* file)
{
	pr_info("Device File Closed...!!!\n");
	return 0;
}

static ssize_t ex_read(struct file* filp, char __user* buf, size_t len,
		loff_t* off)
{
	if (kernel_buffer)
	{
		if( copy_to_user(buf, kernel_buffer, len) > 0)
		{
			pr_err("ERROR: Not all the bytes have been copied to user\n");
		}
	}
	pr_info("Read function : Mem = %s \n", kernel_buffer);

	return (0);
}

static ssize_t ex_write(struct file* filp, const char __user* buf, size_t
		len, loff_t* off)
{

	if( copy_from_user( kernel_buffer, buf, len ) > 0) {
		pr_err("ERROR: Not all the bytes have been copied from user\n");
	}

	pr_info("Write Function : Mem = %s\n", kernel_buffer);

	return len;
}

static int __init ex_driver_init(void)
{
	pr_info("Allocating Major number\n");
	if((alloc_chrdev_region(&dev, 0, 1, "Device")) < 0){
		pr_err("Cannot allocate major number\n");
		unregister_chrdev_region(dev,1);
	}
	pr_info("Major = %d Minor = %d \n",MAJOR(dev), MINOR(dev));

	cdev_init(&ex_cdev,&fops);

	if((cdev_add(&ex_cdev, dev, 1)) < 0){
		pr_err("Cannot add the device to the system\n");
		unregister_chrdev_region(dev,1);
		return -1;
	}

	if((dev_class = class_create(THIS_MODULE,"Device_class")) == NULL){
		pr_err("Cannot create the struct class\n");
		class_destroy(dev_class);
		return -1;
	}

	if((device_create(dev_class,NULL,dev,NULL,"MyLab5Module")) == NULL){
		pr_err( "Cannot create the Device \n");
		device_destroy(dev_class,dev);
		return -1;
	}

	kernel_buffer = kmalloc(mem_size, GFP_KERNEL);

	pr_info("Device Driver Insert\n");

	return (0);
}

static void __exit ex_driver_exit(void)
{
	printk(KERN_INFO "Kernel Module Removed Successfully...\n");

	device_destroy(dev_class,dev);
	class_destroy(dev_class);
	cdev_del(&ex_cdev);
	unregister_chrdev_region(dev, 1);
	pr_info("Device Driver Remove\n");
}

module_init(ex_driver_init);
module_exit(ex_driver_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Semichastnov Konstantin");
MODULE_DESCRIPTION("A simple Driver");
MODULE_VERSION("1.0");
